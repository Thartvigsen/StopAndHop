import torch
from torch.utils import data
import numpy as np
import os

class ExtraSensory(data.Dataset):
    def __init__(self, path_to_data):
        """
        This is a class for loading ExtraSensory datasets.

        Parameters
        ----------
        path_to_data : str
            This path should point to the directory containing train.pt and test.pt
        """
        super(ExtraSensory).__init__()
        X_train, y_train = torch.load(os.path.join(path_to_data, "train.pt"))
        X_test, y_test = torch.load(os.path.join(path_to_data, "test.pt"))

        self.data = X_train + X_test
        self.labels = torch.hstack([y_train, y_test]).squeeze()

        self.train_ix = np.arange(len(X_train))
        self.test_ix = len(X_train) + np.arange(len(X_test))

        self.data_config = {
            "N_FEATURES" : 30 if "walk" in path_to_data else 3, # 30 features for Walking dataset, 3 features for Running dataset
            "N_CLASSES" : 2,
            "nsteps" : 100,
        }

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix]

class SyntheticData(data.Dataset):
    def __init__(self, N=10000, T=10, mode="early_late"):
        """
        Class for a simple and synthetic irregularly-sampled time series dataset

        Parameters
        ----------
        N : int
            The number of time series instances to generate (classes will always be balanced)
        
        T : int
            The number of observations per time series
        
        mode : str
            From what distribution to sample signal locations.
                'early' will sample signal locations from a normal distribution centered at 25% of the timeline
                'late' will sample signal locations from a normal distribution centered at 75% of the timeline
                'early_late' will sample half early and half late signals
                'uniform' will sample signal locations from a uniform distribution spanning the full timeline
        """
        super(SyntheticData, self).__init__()
        self.T = T 
        self.N = N
        if mode == "early":
            self.signal_times = np.random.normal(.25, .1, (N, 1)).clip(0, 1)
        elif mode == "late":
            self.signal_times = np.random.normal(.75, .1, (N, 1)).clip(0, 1)
        elif mode == "uniform":
            self.signal_times = np.random.uniform(0.0, 1.0, (int(N/2), 1))
        elif mode == "early_late":
            early = np.random.normal(.25, .1, (int(N//2), 1)).clip(0, 1)
            late = np.random.normal(.75, .1, (N//2, 1)).clip(0, 1)
            self.signal_times = np.concatenate((early, late))
        self.signal_times = self.signal_times[np.random.choice(self.N, self.N, replace=False)]
        self._N_FEATURES = 1
        self.data, self.labels = self.loadData()

        self.train_ix = np.random.choice(N, N, replace=False)
        self.test_ix = self.train_ix[int(0.8*N):]
        self.train_ix = self.train_ix[:int(0.8*N)]
 
        self.data_config = {
            "N_FEATURES" : 1,
            "N_CLASSES" : 2,
            "nsteps" : T 
        }

    def __getitem__(self, ix): 
        return self.data[ix], self.labels[ix]

    def getVals(self, timesteps, values, mask, nsteps):
        V = values.shape[1]
        new_vals = np.zeros((nsteps, V))
        new_masks = np.zeros((nsteps, V)) # 1 means REAL
        past = np.ones((nsteps, V)) # Time since last observation
        future = np.ones((nsteps, V)) # Time until next observation
        bins = np.round(np.linspace(0+(1./nsteps), 1, nsteps), 3)
        for v in range(V):
            t0 = timesteps[mask[:, v] == 1].numpy()
            v0 = values[mask[:, v] == 1, v].numpy()
            buckets = (np.abs(t0 - (bins[:, None]-(1./(2*nsteps))))).argmin(0) # map all points to their nearest bin centroid
            for n in range(nsteps):
                ix = np.where(buckets == n)
                new_vals[n, v] = np.nanmean(np.take(v0, ix)) # Take mean of all observations in bin
                new_masks[n, v] = len(ix[0]) > 0 # If more than one value was in bin, mask = 1

                below = t0[np.where(t0 <= bins[n])] # Find timesteps less than current timestep
                if len(below) > 0: # If any exist, take diff between current bin and future step
                    past[n, v] = bins[n] - below.max()
                above = t0[np.where(t0 > bins[n])] # Find timesteps greater than current bin
                if len(above) > 0: # If any exist, take diff between future step and current bin
                    future[n, v] = above.min() - bins[n]
        new_vals[np.isnan(new_vals)] = 0.0
        return new_vals.astype(np.float32), new_masks.astype(np.float32), past.astype(np.float32)

    def loadData(self):
        timesteps = np.random.uniform(0, 1, (self.N, self.T-1))
        all_timesteps = np.concatenate((timesteps, self.signal_times), axis=1)

        values = np.zeros_like(timesteps)
        signals = np.concatenate((np.ones((int(self.N/2), 1)), -1*np.ones((int(self.N/2), 1))), axis=0)
        all_values = np.concatenate((values, signals), axis=1)

        sorted_ix = np.argsort(all_timesteps)

        timesteps = np.array([all_timesteps[i][sorted_ix[i]] for i in range(self.N)])
        values = np.array([all_values[i][sorted_ix[i]] for i in range(self.N)])

        labels = np.array([0]*int(self.N/2) + [1]*int(self.N/2))
        timesteps = torch.tensor(timesteps).float()
        values = torch.tensor(values).unsqueeze(2).float()

        # Shuffle
        ix = np.random.choice(self.N, self.N, replace=False)
        self.signal_times = self.signal_times[ix]
        timesteps = timesteps[ix]
        values = values[ix]
        labels = labels[ix]
        masks = torch.ones_like(values).float() # 1 means all values are truly observed

        data = [] 
        signal_times = []
        for i in range(len(values)):
            new_vals, new_masks, past = self.getVals(timesteps[i], values[i], masks[i], nsteps=self.T)
            data.append((new_vals, new_masks, past))
        labels = torch.tensor(labels, dtype=torch.long)
        return data, labels