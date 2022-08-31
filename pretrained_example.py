from ..src.model import StopAndHop
from ..src.utils import computeAUC
import torch
import numpy as np

# Initialize configs
# DO NOT CHANGE FOR THIS EXAMPLE
model_config = {
    "n_layers"      : 1,
    "batch_size"    : 32,
    "n_epochs"      : 2,
    "learning_rate" : 1e-2,
    "nhid"          : 10,
}
data_config = {
    "N_FEATURES" : 41,
    "N_CLASSES" : 2,
    "nsteps" : 49,
}

if __name__ == "__main__":
    # --- set seeds ---
    torch.manual_seed(42)
    np.random.seed(42)

    # --- load precomputed prefix embeddings ---
    prefix_embeddings_train = torch.load("./loaders/physionet_grud_train_loader.pt")
    prefix_embeddings_test = torch.load("./loaders/physionet_grud_test_loader.pt")
        
    # --- initialize Stop&Hop model and optimizers ---
    model = StopAndHop(model_config, data_config, lam=0.0) # Try increasing lam to encourage earlier predictions!
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # --- train ---
    training_loss = []
    for epoch in range(model_config["n_epochs"]):
        total_loss = 0
        for i, (X, y) in enumerate(prefix_embeddings_train):
            logits, _, _ = model(X, epoch=epoch, test=False)
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
    for i, (X, y) in enumerate(prefix_embeddings_test):
        logits, avg_hp, raw_hp = model(X, epoch=0, test=True)
        y_hat = torch.softmax(logits, 1)
        [predictions.append(j.numpy()) for j in y_hat.detach()]
        [labels.append(j.item()) for j in y]
        halting_points.append(avg_hp)

    auc = computeAUC(np.array(predictions).copy(), np.array(labels).copy())
    print(f"AUC: {np.round(auc, 3)}")
    print(f"Average halting point: {np.mean(halting_points).round(3)}")