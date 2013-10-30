"""
This example shows how to decompose EEG signals into source activations with MVARICA, and visualize time varying connectivity.
"""
import scot
import scot.backend.sklearn     # use scikit-learn backend
#import scot.backend.builtin     # use builtin (default) backend

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


# Set up the analysis object
#
# We simply choose a VAR model order of 30, and reduction to 4 components (that's not a lot!).
ws = scot.Workspace(30, reducedim=4, fs=fs, locations=locs)


# Prepare the data
#
# Here we cut segments from 3s to 4s following each trigger out of the EEG. This
# is right in the middle of the motor imagery period.
data = scot.datatools.cut_segments(raweeg, triggers, 3 * fs, 4 * fs)


# Perform CSPVARICA
ws.set_data(data, classes)
ws.do_cspvarica()


# Prepare the data
#
# Here we cut segments from -2s to 8s around each trigger out of the EEG. This
# covers the whole trial
data = scot.datatools.cut_segments(raweeg, triggers, -2 * fs, 8 * fs)


# Connectivity Analysis
#
# Extract the full frequency directed transfer function (ffDTF) from the
# activations of each class and plot them with matplotlib.
ws.set_data(data, classes, time_offset=-2)
figs = ws.plot_tf_connectivity('ffDTF', 1 * fs, int(0.2 * fs), freq_range=[0, 30], crange=[0,30])

figs['hand'].savefig('hand.png', dpi=900)
figs['foot'].savefig('foot.png', dpi=900)


ws.show_plots()
