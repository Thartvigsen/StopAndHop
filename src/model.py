import torch
from src.utils import *
from torch import nn
from src.modules import *

def getTimesEmbeddings(halting_points, embedding_times):
    # Convert halting points (indices) to the true halting times associated with the corresponding prefix embeddings
    # halting_points is of shape (batch_size,) embedding_times is of shape (batch_size, num_embedding_timesteps)

    # Some true embedding times are set to 0 if the time series ended earlier, so reassign them to max
    embedding_times[:, 1:] = torch.where(embedding_times[:, 1:] == 0.0, embedding_times[:, 1:].max(1, keepdim=True)[0], embedding_times[:, 1:])
    return embedding_times[range(len(halting_points)), halting_points.long()]

def getTimes(halting_points, embedding_times):
    # Convert halting points (indices) to the true halting times associated with the corresponding prefix embeddings
    # halting_points is of shape (batch_size,) embedding_times is of shape (batch_size, num_embedding_timesteps)

    # Some true embedding times are set to 0 if the time series ended earlier, so reassign them to max
    embedding_times[:, 1:] = torch.where(embedding_times[:, 1:] == 0.0, embedding_times[:, 1:].max(1, keepdim=True)[0], embedding_times[:, 1:])
    return embedding_times[range(len(halting_points)), halting_points.long()]

class Model(nn.Module):
    def __init__(self, config, data_setting):
        super(Model, self).__init__()

        # --- Model hyperparameters ---
        self._nepoch = config["n_epochs"]
        self.nlayers = config["n_layers"]
        self.bsz = config["batch_size"]

        # --- data setting ---
        self._ninp = data_setting["N_FEATURES"]
        self._nfeatures = data_setting["N_FEATURES"]
        self._nclasses = data_setting["N_CLASSES"]

    def unpack(self, data):
        """data comes from the loader and contains 3 things"""
        vals, masks, past = data
        vals = vals.transpose(0, 1).float() # Convert to shape T x B x V
        masks = masks.transpose(0, 1).float()
        past = past.transpose(0, 1).float()
        past[torch.isnan(past)] = 0.0
        vals[torch.isnan(vals)] = 0.0
        masks[torch.isnan(masks)] = 1.0
        past[torch.isnan(past)] = 1.0
        return vals, masks, past 

    def initHidden(self, B):
        hidden = (torch.zeros((self._nlayers, B, self.nhid)),
                  torch.zeros((self._nlayers, B, self.nhid)))
        return hidden

class StopAndHopPretrained(Model):
    def __init__(self, config, data_setting, std=0.1, lam=0.0):
        super(StopAndHopPretrained, self).__init__(config, data_setting)
        tmax = data_setting["nsteps"]
        self.std = std
        self.lam = lam
        self.nhid = config["nhid"] # dimension of the prefix embeddings, hard-coded since they are precomputed
        self.HaltingPolicy = HaltingPolicy(config, self.nhid+self._nclasses, tmax, self.bsz, self.std, lam)
    
    def forward(self, X, epoch, test):
        if test:
            self.HaltingPolicy._eps = 0.0
        else:
            self.HaltingPolicy._eps = self.HaltingPolicy._exponentials[epoch]

        prefix_embeddings, model_predictions, times = X
        B, T, V = prefix_embeddings.shape
        self.HaltingPolicy.initLoggers()
        self.HaltingPolicy.checkvec = 0*torch.ones((B, 1))
        self.HaltingPolicy.t_max = torch.ones((B))*T

        predictions = -torch.ones((self.bsz, self._nclasses), requires_grad=True)
        halt_points = -torch.ones((self.bsz))
        for t in range(T):
            hp_in = torch.cat((prefix_embeddings[:, t, :], model_predictions[:, t, :]), dim=-1)
            t_in = torch.ones((B, 1))*(t/T)
            halt_decision = self.HaltingPolicy(hp_in, torch.ones_like(t_in)*t, halt_points)
            predictions = torch.where((halt_decision == 1) & (predictions == -1), model_predictions[:, t, :], predictions) #  stop and hasn't stopped yet
            halt_points = torch.where((halt_decision.squeeze() == 1) & (halt_points == -1), torch.ones_like(halt_points)*(t+1), halt_points)
            if (halt_points == -1).sum() == 0:
                break
        model_predictions = torch.where(predictions == -1, model_predictions[:, -1, :], predictions).squeeze()#.detach()
        halt_points = torch.where(halt_points == -1, torch.ones_like(halt_points)*T, halt_points).squeeze(0)#.clamp(0, 1)
        self.halt_points = halt_points
        out_times = getTimesEmbeddings(halt_points-1, times)
        return model_predictions, out_times.mean(), out_times/T
    
    def computeLoss(self, model_predictions, labels):
        return self.HaltingPolicy.computeLoss(model_predictions, labels, self.halt_points)

class StopAndHop(Model):
    def __init__(self, config, data_setting, std=0.1, lam=0.0):
        super(StopAndHop, self).__init__(config, data_setting)
        tmax = data_setting["nsteps"]
        self.std = std
        self.lam = lam
        self.nhid = config["nhid"] # dimension of the prefix embeddings, hard-coded since they are precomputed
        self.HaltingPolicy = HaltingPolicy(config, self.nhid+self._nclasses, tmax, self.bsz, self.std, lam)
        self.Classifier = createNet(self.nhid, self._nclasses)

        # --- initialize GRU updates
        combined_dim = self.nhid + 2*self._ninp # Input and missingness vector
        self._zeros_x = torch.zeros(self._ninp)
        self._zeros_h = torch.zeros(self.nhid)
        self._h_grads = []

        # --- mappings ---
        self.z = nn.Linear(combined_dim, self.nhid) # Update gate
        self.r = nn.Linear(combined_dim, self.nhid) # Reset gate
        self.h = nn.Linear(combined_dim, self.nhid)
        #self.gamma_x = FilterLinear(self._ninp, self._ninp)
        self.gamma_x = createNet(self._ninp, self._ninp)
        self.gamma_h = createNet(self._ninp, self.nhid)

    def decay_update(self, x, h, m, dt, x_prime, x_mean):
            # --- compute decays ---
            delta_x = torch.exp(-torch.max(self._zeros_x, self.gamma_x(dt)))

            # --- apply state-decay ---
            delta_h = torch.exp(-torch.max(self._zeros_h, self.gamma_h(dt)))
            h = delta_h * h 

            x_prime = m*x + (1-m)*x_prime # Update last-observed value

            # --- estimate new x value ---
            x = m*x + (1-m)*(delta_x*x_prime + (1-delta_x)*x_mean)

            # --- gating functions ---
            combined = torch.cat((x, h, m), dim=1)
            r = torch.sigmoid(self.r(combined))
            z = torch.sigmoid(self.z(combined))
            new_combined = torch.cat((x, torch.mul(r, h), m), dim=1)
            h_tilde = torch.tanh(self.h(new_combined))
            h = (1 - z)*h + z*h_tilde
            return h, x_prime
    
    def forward(self, data, test=False, epoch=0):
        if test:
            self.HaltingPolicy._eps = 0.0
        else:
            self.HaltingPolicy._eps = self.HaltingPolicy._exponentials[epoch]

        vals, masks, past = self.unpack(data)
        T, B, V = vals.shape
        self.HaltingPolicy.initLoggers()
        self.HaltingPolicy.checkvec = 0*torch.ones((B, 1))
        self.HaltingPolicy.t_max = torch.ones((B))*T

        h = torch.zeros(B, self.nhid) # Initialize hidden state as 0s
        x_prime = torch.zeros(self._ninp) # Initialize estimated x values as 0s
        predictions = -torch.ones((self.bsz, self._nclasses), requires_grad=True)
        halt_points = -torch.ones((self.bsz))
        t = 0
        while t < T:
            x, m, d = vals[t], masks[t], past[t] # extract values for current timestep
            x_mean = (masks[:(t+1)]*vals[:(t+1)]).sum(0)/masks[:(t+1)].sum(0)
            x_mean[torch.isnan(x_mean)] = 0.0
            h, x_prime = self.decay_update(x, h, m, d, x_prime, x_mean) # Compute new RNN hidden state
            logits = self.Classifier(h) # Compute logits for each element of the batch
            hp_in = torch.cat((h, logits), dim=1).detach()
            t_in = torch.ones((B, 1))*(t/T)
            halt_decision = self.HaltingPolicy(hp_in, torch.ones_like(t_in)*t, halt_points)
            predictions = torch.where((halt_decision == 1) & (predictions == -1), logits, predictions)
            halt_points = torch.where((halt_decision.squeeze() == 1) & (halt_points == -1), torch.ones_like(halt_points)*(t+1), halt_points)
            if (halt_points == -1).sum() == 0: # If all elements in the batch have been stopped, no need to continue looping
                break
            else:
                t += 1
        logits = torch.where(predictions == -1, logits, predictions).squeeze()
        halt_points = torch.where(halt_points == -1, torch.ones_like(halt_points)*T, halt_points).squeeze(0)
        self.halt_points = halt_points
        out_times = getTimes(halt_points-1, past.cumsum(0)[:, :, 0].T)
        return logits, out_times.mean(), out_times/T
    
    def computeLoss(self, logits, labels):
        return 10*F.cross_entropy(logits, labels) + self.HaltingPolicy.computeLoss(logits, labels, self.halt_points)
