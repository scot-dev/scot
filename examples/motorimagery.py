"""
Load Motor Imagery example data
"""
import numpy as np
from os.path import abspath, dirname, join
from scot.matfiles import loadmat
from eegtopo.eegpos3d import positions as eeg_locations


def __load():
    matfile = loadmat(join(abspath(dirname(__file__)), 'motorimagery.mat'))['s0']

    class Data: pass

    Data.description = """
                       The Data set contains a continuous 45 channel EEG recording of a motor imagery
                       experiment. The Data was preprocessed to reduce eye movement artifacts and
                       resampled to a sampling rate of 100 Hz.
                       With a visual cue the subject was instructed to perform either hand of foot
                       motor imagery. The the trigger time points of the cues are stored in 'tr', and
                       'cl' contains the class labels (hand: 1, foot: -1). Duration of the motor 
                       imagery period was approximately 6 seconds.
                       """

    Data.samplerate = matfile['fs'] # Sampling rate
    Data.num_trials = matfile['T']  # Number of trials
    Data.triggers = matfile['tr']   # Trigger locations
    Data.eeg = matfile['eeg']       # EEG Data

    # Class labels
    cltrans = { 1 : 'hand', -1 : 'foot' }
    Data.classes = np.array([cltrans[c] for c in matfile['cl']])

    # Unfortunately, the EEG channel labels are not stored in the file, so we set them manually.
    Data.labels = ['AF7', 'AFz', 'AF8', 'F3', 'F1',
                   'Fz', 'F2', 'F4', 'FT7', 'FC5',
                   'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                   'FC6', 'FT8', 'C5', 'C3', 'C1',
                   'Cz', 'C2', 'C4', 'C6', 'CP5',
                   'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                   'CP6', 'P7', 'P3', 'Pz', 'P4',
                   'P8', 'PO3', 'POz', 'PO4', 'O1',
                   'Oz', 'O2', 'O9', 'Iz', 'O10']

    # Obtain default electrode locations corresponding to the channels
    Data.locations = [[v for v in eeg_locations[l].vector] for l in Data.labels]

    return Data


data = __load()

