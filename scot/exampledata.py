# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2016 SCoT Development Team

from os import makedirs
from os.path import expanduser, isfile, isdir, join
from requests import get  # this is a mandatory dependency because urllib sucks
import numpy as np

from .matfiles import loadmat
from .eegtopo.eegpos3d import positions
from . import config


default_storage = expanduser(config.get("scot", "data"))
valid_datasets = ["mi"]
files = {"mi": ["motorimagery.mat"]}
url = {"mi": "https://github.com/scot-dev/scot-data/raw/master/scotdata/"}


def load(dataset="mi", datadir=default_storage):
    """Load example dataset.

    If the requested dataset is not found in the location specified by
    `datadir`, the function attempts to download it.

    Parameters
    ----------
    dataset : str
        Which dataset to load. Currently only 'mi' is supported.
    datadir : str
        Path to the storage location of example datasets. Datasets are
        downloaded to this location if they cannot be found. If the directory
        does not exist it is created.

    Returns
    -------
        data : list of dicts
            The data set is stored in a list, where each list element
            corresponds to data from one subject. Each list element is a
            dictionary with the following keys:
              "eeg" ... EEG signals
              "triggers" ... Trigger latencies
              "labels" ... Class labels
              "fs" ... Sample rate
              "locations" ... Channel locations
    """
    if dataset not in valid_datasets:
        raise ValueError("Example data '{}' not available.".format(dataset))
    if not isdir(datadir):
        makedirs(datadir)

    data = []

    for file in files[dataset]:
        fullfile = join(datadir, file)
        if not isfile(fullfile):
            with open(fullfile, "wb") as f:
                response = get(join(url[dataset], file))
                f.write(response.content)
        data.append(convert_mi(loadmat(fullfile)))

    return data


def convert_mi(mat):
    mat = mat["s0"]
    
    data = {}
    data["fs"] = mat["fs"]
    data["triggers"] = np.asarray(mat["tr"], dtype=int)
    data["eeg"] = mat["eeg"].T

    cltrans = {1: "hand", -1: "foot"}
    data["labels"] = np.array([cltrans[c] for c in mat["cl"]])

    # Set EEG channel labels manually
    labels = ["AF7", "AFz", "AF8", "F3", "F1", "Fz", "F2", "F4", "FT7", "FC5",
              "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "C5", "C3",
              "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2",
              "CP4", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3", "POz", "PO4",
              "O1", "Oz", "O2", "O9", "Iz", "O10"]
    data["locations"] = [[v for v in positions[l].vector] for l in labels]
    return data
