"""
This example shows how to decompose EEG signals into source activations with
MVARICA, and visualize a connectivity.
"""

import scot.backend.sklearn     # use builtin (default) backend
import scot

# The example data set contains a continuous 45 channel EEG recording of a motor
# imagery experiment. The data was preprocessed to reduce eye movement artifacts
# and resampled to a sampling rate of 100 Hz.
# With a visual cue the subject was instructed to perform either hand of foot
# motor imagery. The the trigger time points of the cues are stored in 'tr', and
# 'cl' contains the class labels (hand: 1, foot: -1). Duration of the motor
# imagery period was approximately 6 seconds.
from motorimagery import data as midata

raweeg = midata.eeg
triggers = midata.triggers
classes = midata.classes
fs = midata.samplerate
locs = midata.locations


# Prepare the data
#
# Here we cut segments from 3s to 4s following each trigger out of the EEG. This
# is right in the middle of the motor imagery period.
data = scot.datatools.cut_segments(raweeg, triggers, 3 * fs, 4 * fs)


# Set up the analysis object
#
# We simply choose a VAR model order of 35, and reduction to 4 components (that's not a lot!).
ws = scot.Workspace({'model_order': 35}, reducedim=4, fs=fs, locations=locs)


# Perform MVARICA and plot the components
ws.set_data(data, classes)
ws.do_mvarica()

p = ws.var_.test_whiteness(50)
print('Whiteness:', p)

ws.plot_source_topos()


# Connectivity Analysis
#
# Extract the full frequency directed transfer function (ffDTF) from the
# activations of each class and plot them with matplotlib.
ws.fit_var()
ws.plot_connectivity('ffDTF', freq_range=[0, 30])
ws.show_plots()
