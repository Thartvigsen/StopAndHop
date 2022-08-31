from torch import nn
from src.utils import *
from torch.distributions import Normal
import torch
import torch.nn.functional as F

class HaltingPolicy(nn.Module):
    def __init__(self, config, ninp, tmax, B, std, lam=0.0):
        super(HaltingPolicy, self).__init__()
        self._nepoch = config["n_epochs"]
        self.LAMBDA = lam 
        self.tmax = tmax
        self.B = B 
        self.std = std

        # --- initialize exponential decay values to be used during training ---
        self._exponentials = exponentialDecay(self._nepoch)

        # --- initialize checkvec, which will tell us which time series are ready to stop ---
        self.checkvec = torch.ones((B, 1)) 

        # --- Mappings ---
        self.stopNet = createNet(ninp, 2, n_layers=1, n_units=20)
        self.hopNet = createNet(ninp, 1, n_layers=1, n_units=20)
        self.BaselineNetwork = createNet(ninp, 1)
    
    def initLoggers(self):
        self.log_pi_stop = []
        self.log_pi_hop = []
        self.halt_decisions = []
        self.wait_probs = []
        self.halting_steps = []
        self.baselines = []
        self.halt_points = -torch.ones((self.B))
        self.grad_mask = torch.ones((self.B), requires_grad=False)
        self.grad_masks = []
        self.allowable_actions = []
         
    def forward(self, x, t, halt_points):
        """
        Given a prefix embedding, decide whether to Stop or Hop. If Stop, return 1, if Hop, decide for-how-long

        To batch this function, we maintain an indicator variable for each element of the batch, indicating whether
        or not it can be stopped: If its hop size is longer than another batch element, this function will still run
        for the hopping time series.

        The key idea is that we force the model to choose WAIT if it's not actually ready to stop, then simply don't
        update the network's weights based on any decisions made while it wasn't actually ready to stop. We use a
        binary vector called "self.checkvec" to make this process work.
        """
        # --- choose whether or not to stop ---
        v = (t == self.checkvec).long() # Tells us which batch items are ready to go: If 1, allowed to stop
        action, log_pi_stop, wait_probs = self.stopPolicy(x) # Action is an integer
        stop = (action == 0).long().unsqueeze(1)
        stop = v*stop
        grad_mask = torch.where(halt_points > 0.0, torch.zeros((self.B)), torch.ones((self.B))).detach() # for time series that have already been halted, mask out the gradients with 0s
        self.grad_masks.append(grad_mask)
        self.wait_probs.append(wait_probs)
        self.log_pi_stop.append(log_pi_stop)
        
        # --- choose how long to hop for ---
        hop_size, log_pi_hop = self.hopPolicy(x)
        hop_size = (self.tmax*hop_size).long() # round the hop_size
        self.log_pi_hop.append(log_pi_hop)
        
        # --- run baseline network ---
        b = self.BaselineNetwork(x).squeeze()
        self.baselines.append(b)
        
        # --- update the vector that tells us which batch items are ready to stop ---
        self.checkvec = torch.where(self.checkvec==t, t+hop_size, self.checkvec)
        self.allowable_actions.append(v)
        return stop
    
    def stopPolicy(self, x):
        """
        The stop policy (pi_stop in the paper) decides whether or not to stop given a prefix embedding.
        There are 5 steps:
            1. predict probability of halting (probs) with stopNet, a neural network that predicts probabilities of Stop and Hop
            2. save the predicted probability of waiting (wait_probs) to penalize during training
            3. re-assign probabilities according to epsilon-greedy search: Random actions early in training, learned actions late in training
            4. select an action (Stop or Hop) from the multinomial distribution parameterized by probs
            5. save the log probabilities of the chosen actions for training
        """
        probs = torch.softmax(self.stopNet(x), dim=1)
        wait_probs = -torch.log(probs[:, 0]) # compute log probs of not-halt = wait
        probs = (1.0-self._eps)*probs + (self._eps)*torch.FloatTensor([0.05]) # Explore/exploit (can't be 0)
        action = probs.multinomial(num_samples=1).squeeze()
        log_pi = torch.gather(torch.log(probs), 1, action.unsqueeze(1)).squeeze()
        return action, log_pi, wait_probs
    
    def hopPolicy(self, x):
        """
        The hop policy (pi_hop in the paper) decides for how long to wait before trying to halt again.
        There are 3 steps:
            1. predict mean (self.mu) of a normal distribution using hopNet, a neural network that predicts 1 value
            2. Actually sample from the normal distribution (l_t)
            3. Ensure sample is in range [0, 1] by clamping between -1 and 1, then taking the absolute value.

        log_pi is used to train hopNet, so training can proceed even with the non-differentiable clamping and absolute value actions later on
        """
        self.mu = torch.tanh(self.hopNet(x))
        distribution = Normal(self.mu, self.std)
        l_t = distribution.rsample()
        l_t = l_t.detach()
        log_pi = distribution.log_prob(l_t)
        l_t = torch.clamp(l_t, -1, 1)
        l_t = torch.abs(l_t)
        return l_t, log_pi
    
    def getReward(self, logits, labels, halt_points):
        # convert logits to class predictions
        y_hat = torch.softmax(logits.detach(), dim=1)
        y_hat = torch.max(y_hat, 1)[1]

        # get reward
        reward = (2*(y_hat.float().round() == labels.squeeze().float()).float()-1) # +1 for accurate, -1 for inaccurate
        return reward.unsqueeze(1).detach()
    
    def computeLoss(self, logits, labels, halt_points):
        # --- collate lists containing variables collected throughout training ---
        log_pi_stop = torch.stack(self.log_pi_stop).transpose(0, 1) # Batch first
        log_pi_hop = torch.stack(self.log_pi_hop).transpose(0, 1).squeeze(2) # Batch first
        wait_probs = torch.stack(self.wait_probs)
        baselines = torch.stack(self.baselines).transpose(0, 1)
        grad_mask = torch.stack(self.grad_masks).transpose(0, 1).detach() # grad_mask is a binary mask for which series halted when, allowing for batching
        self.allowable_actions = torch.stack(self.allowable_actions)
        
        # --- compute reward ---
        self.r = self.getReward(logits, labels, halt_points)
        self.R = self.r * grad_mask # broadcast reward to all timesteps and set rewards to 0 with grad_mask to avoid training based on timesteps after a series was halted

        # --- rescale reward with baseline ---
        b = grad_mask * baselines # mask out baseline predictions for after-halting timesteps
        self.adjusted_reward = self.R - b.detach() # Adjust reward according to baseline (reduces variance in the policy updates)

        # --- compute losses ---
        self.loss_b = F.mse_loss(b, self.R) # encourage baseline to accurately predict R (approaches mean(R) in the aggregate)
        self.wait_penalty = wait_probs.mean() # penalize large probabilities of waiting
        self.loss_stop = (-log_pi_stop*self.adjusted_reward).sum(1).mean() + self.LAMBDA*wait_probs.mean() # Apply policy update to pi_stop and penalize late predictions
        self.loss_hop = (-log_pi_hop*self.adjusted_reward).sum(1).mean() # Apply policy update to pi_hop to encourage accurate predictions
        loss = self.loss_stop + self.loss_hop + self.loss_b # Add loss components together for final loss
        return loss
