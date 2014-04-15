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
import scotdata.motorimagery as midata

raweeg = midata.eeg
triggers = midata.triggers
classes = midata.classes
fs = midata.samplerate
locs = midata.locations


# Set up the analysis object
#
# We simply choose a VAR model order of 35, and reduction to 4 components (that's not a lot!).
ws = scot.Workspace({'model_order': 35}, reducedim=4, fs=fs, locations=locs)


# Prepare the data
#
# Here we cut segments from 3s to 4s following each trigger out of the EEG. This
# is right in the middle of the motor imagery period.
data = scot.datatools.cut_segments(raweeg, triggers, 3 * fs, 4 * fs)


# Perform CSPVARICA
ws.set_data(data, classes)
ws.do_cspvarica()

p = ws.var_.test_whiteness(50)
print('Whiteness:', p)



# Prepare the data
#
# Here we cut segments from -2s to 8s around each trigger out of the EEG. This
# covers the whole trial
data = scot.datatools.cut_segments(raweeg, triggers, -2 * fs, 8 * fs)


# Configure plotting options
ws.plot_f_range = [0, 30]       # Only show 0 - 30 Hz
ws.plot_diagonal = 'topo'       # Put topo plots on the diagonal
ws.plot_outside_topo = False    # no topo plots above and to the left


# Connectivity Analysis
#
# Extract the full frequency directed transfer function (ffDTF) from the
# activations of each class and plot them with matplotlib.
ws.set_data(data, classes, time_offset=-1)

fig = ws.plot_connectivity_topos()
ws.set_used_labels(['hand'])
ws.get_tf_connectivity('ffDTF', 1 * fs, int(0.2 * fs), plot=fig, crange=[0,30])
fig.suptitle('Hand')

fig = ws.plot_connectivity_topos()
ws.set_used_labels(['foot'])
ws.get_tf_connectivity('ffDTF', 1 * fs, int(0.2 * fs), plot=fig, crange=[0,30])
fig.suptitle('Foot')

ws.show_plots()
