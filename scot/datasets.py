# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2016 SCoT Development Team

from os import makedirs
from os.path import expanduser, isfile, isdir, join
from requests import get
import numpy as np
import hashlib

from .matfiles import loadmat
from .eegtopo.eegpos3d import positions
from . import config


datadir = expanduser(config.get("scot", "data"))
datasets = {"mi": {"files": ["motorimagery.mat"], "md5": ["239a20a672f9f312e9d762daf3adf214"],
                   "url": "https://github.com/scot-dev/scot-data/raw/master/scotdata/"}}


def fetch(dataset="mi", datadir=datadir):
    """Fetch example dataset.

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
    if dataset not in datasets:
        raise ValueError("Example data '{}' not available.".format(dataset))
    else:
        files = datasets[dataset]["files"]
        url = datasets[dataset]["url"]
        md5 = datasets[dataset]["md5"]
    if not isdir(datadir):
        makedirs(datadir)

    data = []

    for n, filename in enumerate(files):
        fullfile = join(datadir, filename)
        if not isfile(fullfile):
            with open(fullfile, "wb") as f:
                response = get(join(url, filename))
                f.write(response.content)
        with open(fullfile, "rb") as f:  # check if MD5 of downloaded file matches original hash
            hash = hashlib.md5(f.read()).hexdigest()
        if hash != md5[n]:
            raise MD5MismatchError("MD5 hash of {} does not match {}.".format(fullfile, md5[n]))
        data.append(convert(dataset, loadmat(fullfile)))

    return data


def convert(dataset, mat):
    if dataset == "mi":
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


class MD5MismatchError(Exception):
    pass
