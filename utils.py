import torch
from torch import nn
import numpy as np
import csv
import os.path
import os
from itertools import product

class MainWriter(object):
    def __init__(self):
        self.header = ("from expConfig import *\n"
                       "from model import *\n"
                       "from dataset import *\n"
                       "from metric import *\n"
                       "from utils import ConfigReader\n"
                       "import argparse\n\n"
        
                       "parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)\n"
                       "parser.add_argument('--taskid', type=int, default=0, help='the experiment task to run')\n"
                       "args = parser.parse_args()\n\n"
        
                       "# parse parameters\n"
                       "t = args.taskid\n\n"
        
                       "c_reader = ConfigReader()\n\n")
        
        # --- set experimental configs ---
        self.datasets = [
            #"""SimpleSignalEarlyLate()""",
            #"""PNLoad(config=config)""",
            #"""ExtraSensoryUser(label_name='label:FIX_walking', threshold=0.001, width=200, nref=20)""",
            """ExtraSensoryUser(label_name='label:FIX_running', threshold=0.001, width=100, nref=100)""",
            #"""ExtraSensoryUser(label_name='label:SLEEPING', threshold=0.001, width=100, nref=10)""",
            #"""ExtraSensoryUser(label_name='label:LYING_DOWN', threshold=0.001, width=100, nref=10)""",
            #"""IHM(config=config)""",
        ]

        self.metrics = """[Accuracy(), AUC_macro()]"""

        self.models = [
            #"""Pretrain(config, d.data_setting, estimator_name="last", lam=0.0)""",
            #"""PretrainGRUD(config, d.data_setting, halt_prop={})""",
            #"""GRUMeanThresh(config, d.data_setting, thresh={})""",
            #"""GRUD(config, d.data_setting, halt_prop={})""",
            #"""GRUMean(config, d.data_setting, halt_prop={})""",
            #"""AdaptiveGRUD(config, d.data_setting, options=[5], lam={})""",
            #"""AdaptiveGRUD(config, d.data_setting, options=[20], lam={})""",
            """AdaptiveGRUD(config, d.data_setting, options=[1], lam={})""",
            #"""AdaptiveImpute(config, d.data_setting, options=[2], lam={})""",
            #"""AdaptiveImpute(config, d.data_setting, options=[1,2], lam={})""",
            #"""AdaptiveImpute(config, d.data_setting, options=[1,2,5], lam={})""",
            #"""AdaptiveGRUD(config, d.data_setting, options=[1], lam={})""",
            #"""AdaptiveGRUD(config, d.data_setting, options=[1,2], lam={})""",
            #"""AdaptiveGRUD(config, d.data_setting, options=[1,2,5], lam={})""",
            #"""AdaptiveGRUD(config, d.data_setting, options=[5], lam={})""",
            #"""AdaptiveGRUD(config, d.data_setting, options=[5, 10], lam={})""",
            #"""AdaptiveGRUD(config, d.data_setting, options=[2, 5, 10], lam={})""",
            #"""AdaptiveGRUDLookahead(config, d.data_setting, options=[1], lam={})""",
            #"""AdaptiveGRUD(config, d.data_setting, options=[5, 10], lam={})""",
            #"""AdaptiveGRUDLookahead(config, d.data_setting, options=[5, 10], lam={})""",
        ]

        #self.alphas = [0.0, 1.0]
        #self.alphas = [0.0, 1e-01, 2e-01, 3e-01, 4e-1, 5e-1, 1e-02, 2e-02, 3e-02,
        #               1e-03, 1e-04, 2e-03, 2e-04, 3e-03, 3e-04]
        #self.alphas = [1.0]
        #self.alphas = [0.0, 1e-01, 2e-01, 3e-01, 1e-02, 2e-02, 3e-02,
        #               1e-03, 1e-04, 1e-05, 1e-06, 2e-03, 2e-04, 2e-05,
        #               2e-06, 3e-03, 3e-04, 3e-05, 3e-06]
        self.alphas = logspace(np.arange(1, 5), start=1, end=5)
        self.alphas = np.concatenate((np.zeros(1), self.alphas))
        #self.alphas = np.round(np.linspace(0.0, 0.5, 21), 3)
        #self.alphas = np.round(np.linspace(0.36, 0.38, 21), 3)
        #self.alphas = np.round(np.linspace(0.0, 1.0, 11), 3)
        #self.alphas = np.round(np.linspace(0.1, 1.0, 10), 3)
        #self.alphas = np.round(np.linspace(0.5, 1.0, 21), 3)
        #self.alphas = [0.0]
        #self.alphas = [10]

        #self.alphas = np.linspace(1, 10, 11).astype(np.int32)

        #self.alphas = np.linspace(5, 100, 20).astype(np.int32)
        #self.weight_decays = [1e-5, 1e-4, 1e-3]
        #self.learning_rates = [1e-4, 1e-3, 1e-2]
        self.weight_decays = [1e-5]
        self.learning_rates = [1e-4, 1e-3, 1e-2]
        #self.weight_decays = [1e-02, 1e-03, 1e-04, 1e-05]
        #self.alphas = np.linspace(5, 90, 18).astype(np.int32)

        self.n_iterations = 5
        #"    #torch.save(p.model.state_dict(), './models/physionet/grud.pt')"

    def write(self):
        t = 0
        for d in self.datasets:
            for model in self.models:
                for a in self.alphas:
                    for w in self.weight_decays:
                        for l in self.learning_rates:
                            for i in range(self.n_iterations):
                                text = ("if t == {0}:\n"
                                        "    # --- Iteration: {1} ---\n"
                                        "    np.random.seed(t%{7})\n"
                                        "    config = c_reader.read(t)\n"
                                        "    d = {2}\n"
                                        "    m = {3}\n"
                                        "    e = {4}\n"
                                        "    p = ExpConfig(d=d,\n"
                                        "                  m=m,\n"
                                        "                  e=e,\n"
                                        "                  config=config,\n"
                                        "                  weight_decay={5},\n"
                                        "                  learning_rate={6},\n"
                                        "                  iteration=t%{7})\n"
                                        "    p.run()\n\n".format(t,
                                                                 i+1,
                                                                 d,
                                                                 model.format(a),
                                                                 self.metrics,
                                                                 w,
                                                                 l,
                                                                 self.n_iterations))
                                self.header += text
                                t += 1

        with open("main.py", "w") as f:
            f.write(self.header)
            f.close()

def squaredExponentialKernel(r, t, alpha=100):
    dist = torch.exp(torch.mul(-alpha, 1*torch.sub(r, t).pow(2)))
    #mask = torch.zeros_like(t)
    #mask[t > 0] = 1 #
    return dist.sum(1)#*mask# + 1e-07 # If dist goes to 0, still need to divide

def logspace(base, start, end):
    outs = []
    for b in base:
        for i in range(1, end-start+2):
            outs.append(float("{}e-{}".format(b, i)))
    return np.array(outs)

def normalize_data(data):
    reshaped = data.reshape(-1, data.size(-1))

    att_min = torch.min(reshaped, 0)[0]
    att_max = torch.max(reshaped, 0)[0]

    # we don't want to divide by zero
    att_max[ att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm, att_min, att_max

def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def add_mask(data_dict):
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]

    if mask is None:
        mask = torch.ones_like(data).to(get_device(data))

    data_dict["observed_mask"] = mask 
    return data_dict

def split_and_subsample_batch(data_dict, extrap=False, sample_tp=None, cut_tp=None, data_type = "train"):
    if data_type == "train":
        # Training set
        if extrap:
            processed_dict = split_data_extrap(data_dict, dataset = args.dataset)
        else:
            processed_dict = split_data_interp(data_dict)

    else:
        # Test set
        if extrap:
            processed_dict = split_data_extrap(data_dict, dataset = args.dataset)
        else:
            processed_dict = split_data_interp(data_dict)

    # add mask
    processed_dict = add_mask(processed_dict)

    # Subsample points or cut out the whole section of the timeline
    if (sample_tp is not None) or (cut_tp is not None):
        processed_dict = subsample_observed_data(processed_dict,
            n_tp_to_sample = args.sample_tp,
            n_points_to_cut = args.cut_tp)

    # if (args.sample_tp is not None):
    #     processed_dict = subsample_observed_data(processed_dict, 
    #         n_tp_to_sample = args.sample_tp)
    return processed_dict

def split_data_extrap(data_dict, dataset = ""):
    # Divides timeline in 2
    device = get_device(data_dict["data"])

    n_observed_tp = data_dict["data"].size(1) // 2
    if dataset == "hopper":
        n_observed_tp = data_dict["data"].size(1) // 3

    split_dict = {"observed_data": data_dict["data"][:,:n_observed_tp,:].clone(),
                  "observed_tp": data_dict["time_steps"][:n_observed_tp].clone(),
                  "data_to_predict": data_dict["data"][:,n_observed_tp:,:].clone(),
                  "tp_to_predict": data_dict["time_steps"][n_observed_tp:].clone()}

    split_dict["observed_mask"] = None
    split_dict["mask_predicted_data"] = None
    split_dict["labels"] = None

    if ("mask" in data_dict) and (data_dict["mask"] is not None):
        split_dict["observed_mask"] = data_dict["mask"][:, :n_observed_tp].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"][:, n_observed_tp:].clone()

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        split_dict["labels"] = data_dict["labels"].clone()

    split_dict["mode"] = "extrap"
    return split_dict

def subsample_observed_data(data_dict, n_tp_to_sample = None, n_points_to_cut = None):
    # n_tp_to_sample -- if not None, randomly subsample the time points. The resulting timeline has n_tp_to_sample points
    # n_points_to_cut -- if not None, cut out consecutive points on the timeline.  The resulting timeline has (N - n_points_to_cut) points
    if n_tp_to_sample is not None:
        # Randomly subsample time points
        data, time_steps, mask = subsample_timepoints(
            data_dict["observed_data"].clone(), 
            time_steps = data_dict["observed_tp"].clone(), 
            mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
            n_tp_to_sample = n_tp_to_sample)

    if n_points_to_cut is not None:
        # Remove consecutive time points
        data, time_steps, mask = cut_out_timepoints(
            data_dict["observed_data"].clone(), 
            time_steps = data_dict["observed_tp"].clone(), 
            mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
            n_points_to_cut = n_points_to_cut)

    new_data_dict = {}
    for key in data_dict.keys():
        new_data_dict[key] = data_dict[key]

    new_data_dict["observed_data"] = data.clone()
    new_data_dict["observed_tp"] = time_steps.clone()
    new_data_dict["observed_mask"] = mask.clone()

    if n_points_to_cut is not None:
        # Cut the section in the data to predict as well
        # Used only for the demo on the periodic function
        new_data_dict["data_to_predict"] = data.clone()
        new_data_dict["tp_to_predict"] = time_steps.clone()
        new_data_dict["mask_predicted_data"] = mask.clone()

    return new_data_dict

def split_data_interp(data_dict):
    device = get_device(data_dict["data"])

    split_dict = {"observed_data": data_dict["data"].clone(),
                "observed_tp": data_dict["time_steps"].clone(),
                "data_to_predict": data_dict["data"].clone(),
                "tp_to_predict": data_dict["time_steps"].clone()}

    split_dict["observed_mask"] = None
    split_dict["mask_predicted_data"] = None
    split_dict["labels"] = None

    if "mask" in data_dict and data_dict["mask"] is not None:
        split_dict["observed_mask"] = data_dict["mask"].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"].clone()

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        split_dict["labels"] = data_dict["labels"].clone()

    split_dict["mode"] = "interp"
    return split_dict

def getFirstIndex(arr, condition):
    tmp = ((condition).long() * (torch.arange(arr.shape[1], 0, -1).unsqueeze(0))).float()
    tmp[:, -1] += 0.1
    tmp = tmp.argmax(1, keepdim=True)
    return tmp

def getLastIndex(arr, condition):
    tmp = ((condition).long() * (torch.arange(0, arr.shape[1]) .unsqueeze(0).unsqueeze(2))).float()
    tmp[:, 0, :] += 0.1
    return tmp.argmax(1, keepdim=True)

def attrToString(obj, prefix,
                 exclude_list=["NAME", "name", "desc", "training", "bsz",
                               "intensities", "train_dataloader", "signal_times",
                               "train_dataset", "test_dataloader", "train_data", "test_data",
                               "test_dataset", "train_labels", "test_labels",
                               "test_ix", "train_ix", "values", "timesteps", "bsz",
                               "deltas", "masks", "epsilons", "val_ix", "r", "T", "N",
                               "Locator", "HaltingPolicy", "Estimator", "RNN", "SCHEDULE_LR", "BATCH_SIZE",
                               "BaselineNetwork", "nlayers", "M", "table", "t0", "stop",
                               "device", "lengths", "ids", "values", "masks", "DataModel",
                               "timesteps", "signal_start", "signal_end",
                               "data", "labels", "signal_locs", "round",
                               "train", "test", "train_labels", "test_labels",
                               "data_setting", "y_train", "y_test", "seq_length"]):
    """Convert the attributes of an object into a unique string of
    path for result log and model checkpoint saving. The private
    attributes (starting with '_', e.g., '_attr') and the attributes
    in the `exclude_list` will be excluded from the string.
    Args:
        obj: the object to extract the attribute values prefix: the
        prefix of the string (e.g., MODEL, DATASET) exclude_list:
        the list of attributes to be exclude/ignored in the
        string Returns: a unique string of path with the
        attribute-value pairs of the input object
    """
    out_dir = prefix #+"-"#+obj.name
    for k, v in obj.__dict__.items():
        if not k.startswith('_') and k not in exclude_list:
            out_dir += "/{}-{}".format(k, ",".join([str(i) for i in v]) if type(v) == list else v)
    return out_dir

def writeCSVRow(row, name, path="./", round=False):
    """
    Given a row, rely on the filename variable to write
    a new row of experimental results into a log file

    New Idea: Write a header for the csv so that I can
    clearly understand what is going on in the file

    Parameters
    ----------
    row : list
        A list of variables to be logged
    name : str
        The name of the file
    path : str
        The location to store the file
    """

    if round:
        row = [np.round(i, 2) for i in row]
    f = path + name + ".csv"
    with open(f, "a+") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(row)

def exponentialDecay(N):
    tau = 1
    tmax = 7
    t = np.linspace(0, tmax, N)
    y = torch.tensor(np.exp(-t/tau), dtype=torch.float)
    return y#/10.

class ConfigReader(object):
    def __init__(self):
        self.path = "/home/twhartvigsen/NewECISTS/configs/"

    def read(self, t):
        """Read config file t as a dictionary.
        If the requested file does not exist, load the base
        configuration file instead.
        """
        if os.path.isfile(self.path+"input_{}.txt".format(t)):
            s = open(self.path+"input_{}.txt".format(t), "r").read()
        else:
            print("Loading base config file.")
            s = open(self.path+"base_config.txt", "r").read()
        return eval(s)

def makedir(dirname):
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print("Made dir")
    except:
        pass

def createRNN(ninp, nhid=100, cell_type="GRU", nlayers=1):
    if cell_type == "GRU":
        return torch.nn.GRU(ninp, nhid, num_layers=nlayers)
    elif cell_type == "LSTM":
        return torch.nn.LSTM(ninp, nhid, num_layers=nlayers)
    else:
        return torch.nn.RNN(ninp, nhid, num_layers=nlayers)

def createNet(n_inputs, n_outputs, n_layers=0, n_units=100, nonlinear=nn.Tanh):
    if n_layers == 0:
        return nn.Linear(n_inputs, n_outputs)
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers-1):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))
        layers.append(nn.Dropout(p=0.5))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)

def getEstimator(estimator_name):
    if estimator_name == "mean":
        return MeanEstimator
    elif estimator_name == "moving_average":
        return MovingAverageEstimator
    elif estimator_name == "last":
        return LastEstimator
    elif estimator_name == "gaps":
        return GapsEstimator
    elif estimator_name == "values":
        return ValuesEstimator
    elif estimator_name == "gaps_values":
        return GapsValuesEstimator
    elif estimator_name == "simple":
        return SimpleEstimator
    elif estimator_name == "decay":
        return DecayEstimator
    elif estimator_name == "gaps_last":
        return GapLastEstimator

def MeanEstimator(data, t, t_prev, *args):
    timesteps, values, masks = data
    B, T = timesteps.shape
    # Assume masks = 1 means it's a real recording
    below_t = (timesteps <= t).long()
    below_t = below_t.unsqueeze(2)*(1-masks) # Set values in the mask to 0 where mask is 0
    x = (below_t*values).sum(1)/below_t.sum(1)
    x[torch.isnan(x)] = 0.0
    return x

def MovingAverageEstimator(data, t, t_prev, *args):
    # Take average in window -- if nothing in window, we need to impute!
    timesteps, values, masks = data
    #print(timesteps)

    #print(t)
    #print(t_prev)
    #print()
    B, T = timesteps.shape
    below_t = timesteps <= t
    above_t_delta = timesteps > t_prev
    in_window = (below_t.unsqueeze(2) & above_t_delta.unsqueeze(2) & (masks == 0)).long()
    x = (in_window*values).sum(1)/in_window.sum(1)
    x[torch.isnan(x)] = 0.
    return x

def SimpleEstimator(data, t, t_prev, *args):
    timesteps, values, masks = data
    B, T = timesteps.shape
    below_t = timesteps <= t
    above_t_delta = timesteps > t_prev
    in_window = (below_t.unsqueeze(2) & above_t_delta.unsqueeze(2) & (masks == 0)).long()
    x = (in_window*values).sum(1)/in_window.sum(1)
    x[torch.isnan(x)] = 0.
    # now compute whiich x vals are 0
    m = (x == 0).long()
    x = torch.cat((x, m), dim=1)
    return x

def LastEstimator(data, t, t_prev, *args):
    timesteps, values, masks = data
    B, T = timesteps.shape
    boosted_timesteps = masks*(100*torch.ones_like(masks)) + (1-masks)*timesteps.unsqueeze(2)
    last_ix = ((boosted_timesteps <= t.unsqueeze(1)).long().sum(1)-1).clamp(0, T).unsqueeze(1)
    x = torch.gather(values, 1, last_ix).squeeze(1)
    return x

def GapLastEstimator(data, t, t_prev, *args):
    timesteps, values, masks = data
    B, T = timesteps.shape
    boosted_timesteps = masks*(100*torch.ones_like(masks)) + (1-masks)*timesteps.unsqueeze(2)
    last_ix = ((boosted_timesteps <= t.unsqueeze(2)).long().sum(1)-1).clamp(0, T).unsqueeze(1)
    x = torch.gather(values, 1, last_ix).squeeze(1)
    last_t = torch.gather(timesteps, 1, last_ix.squeeze()).squeeze(1)
    gap = t.repeat(1, values.shape[2])-last_t
    x = torch.cat((x, last_t), dim=1)
    return x

def GapsEstimator(data, t, t_prev, *args):
    timesteps, values, masks = data
    B, T = timesteps.shape
    diffs = timesteps[:, 1:] - timesteps[:, :-1]
    diffs = torch.cat((torch.zeros((B, 1)), diffs), axis=1)
    boosted_timesteps = masks*(100*torch.ones_like(masks)) + (1-masks)*diffs.unsqueeze(2)
    last_ix = ((boosted_timesteps <= t.unsqueeze(2)).long().sum(1)-1).clamp(0, T).unsqueeze(1)
    diffs = diffs.unsqueeze(2)
    x = torch.gather(diffs, 1, last_ix).squeeze(1)
    return x

def ValuesEstimator(data, t, t_prev, *args):
    timesteps, values, masks = data
    B, T = timesteps.shape
    diffs = timesteps[:, 1:] - timesteps[:, :-1]
    diffs = torch.cat((torch.zeros((B, 1)), diffs), axis=1)
    boosted_timesteps = masks*(100*torch.ones_like(masks)) + (1-masks)*diffs.unsqueeze(2)
    last_ix = ((boosted_timesteps <= t.unsqueeze(2)).long().sum(1)-1).clamp(0, T).unsqueeze(1)
    x = torch.gather(diffs, 1, last_ix).squeeze(1)
    return values

class DecayEstimator(torch.nn.Module):
    def __init__(self, ninp, nout):
        super(DecayEstimator, self).__init__()
        self.fc = createNet(ninp, nout)
        #self.fc = createNet(ninp, nout-1)

    def __call__(self, data, t, t_prev, h):
        return self.forward(data, t, t_prev, h)

    def forward(self, data, t, t_prev, h):
        timesteps, values, masks = data
        B, T = timesteps.shape
        # z-score normalization
        values = (values - values.mean(1).unsqueeze(1))/values.std(1).unsqueeze(1)
        values[torch.isnan(values)] = 0.0
        #t = t.unsqueeze(2)

        # Get running mean
        below_t = (timesteps <= t).long()
        below_t = below_t.unsqueeze(2)*(1-masks) # Set values in the mask to 0 where mask is 0
        running_mean = (below_t*values).sum(1)/below_t.sum(1)
        running_mean[torch.isnan(running_mean)] = 0.

        # Get last value
        boosted_timesteps = masks*(100*torch.ones_like(masks)) + (1-masks)*timesteps.unsqueeze(2)
        last_ix = ((boosted_timesteps <= t.unsqueeze(2)).long().sum(1)-1).clamp(0, T).unsqueeze(1)
        last = torch.gather(values, 1, last_ix).squeeze(1)
        last_t = torch.gather(timesteps, 1, last_ix.squeeze()).squeeze(1)

        #gamma = torch.exp(-torch.max(torch.zeros_like(last_t), self.fc(t.repeat(1, values.shape[2])-last_t)))
        # Compute decay value - this is really brute force.. it could be "parameterize a decay function"
        gamma = torch.sigmoid(self.fc(h))
        x = (1-gamma)*last + (gamma)*running_mean
        x = x.squeeze(0)
        below_t = timesteps <= t
        above_t_delta = timesteps > t_prev
        in_window = (below_t.unsqueeze(2) & above_t_delta.unsqueeze(2) & (masks == 0)).long()
        m = (in_window.sum(1) == 0).long()
        x = torch.cat((x, m), dim=1)
        return x
