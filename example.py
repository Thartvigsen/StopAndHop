from src.model import StopAndHop
from src.utils import computeAUC
from src.dataset import ExtraSensory, SyntheticData#, ExtraSensoryRunning
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np

# Initialize configs
model_config = {
    "n_layers"      : 1, # Number of hidden layers in RNN
    "batch_size"    : 32,
    "n_epochs"      : 20, # Number of training epochs
    "learning_rate" : 1e-2, # Learning rate for optimizer
    "nhid"          : 10, # Dimensionality of the RNN's hidden state
}

if __name__ == "__main__":
    # --- set seeds ---
    torch.manual_seed(42)
    np.random.seed(42)

    # --- load precomputed prefix embeddings ---
    # data = ExtraSensory("./data/walking/") # Walking dataset
    data = ExtraSensory("./data/running/") # Running dataset
    # data = SyntheticData()
    data_config = data.data_config

    # --- get data loaders ---
    train_sampler = SubsetRandomSampler(data.train_ix)
    test_sampler = SubsetRandomSampler(data.test_ix)
    train_loader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=model_config["batch_size"], drop_last=True)
    test_loader = torch.utils.data.DataLoader(data, sampler=test_sampler, batch_size=model_config["batch_size"], drop_last=True)
        
    # --- initialize Stop&Hop model and optimizers ---
    model = StopAndHop(model_config, data_config, lam=0.0) # Try increasing lam to encourage earlier predictions!
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # --- train ---
    training_loss = []
    for epoch in range(model_config["n_epochs"]):
        total_loss = 0
        for i, (X, y) in enumerate(train_loader):
            logits, _, _ = model(X, epoch=epoch)
            loss = model.computeLoss(logits, y)
            if len(logits.shape) > 2:
                logits = logits[-1]
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f'Epoch [{epoch+1}/{model_config["n_epochs"]}]')

    # --- test ---
    predictions = []
    labels = []
    halting_points = []
    for i, (X, y) in enumerate(test_loader):
        logits, avg_hp, raw_hp = model(X, test=True)
        y_hat = torch.softmax(logits, 1)
        [predictions.append(j.numpy()) for j in y_hat.detach()]
        [labels.append(j.item()) for j in y]
        halting_points.append(avg_hp)

    auc = computeAUC(np.array(predictions).copy(), np.array(labels).copy())
    print(f"Test AUC: {np.round(auc, 3)}")
    print(f"Average test halting point: {np.round(np.mean(halting_points), 3)}")