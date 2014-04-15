"""
This example shows how to create surrogate connectivity to determine
if connectivity is statistically significant.
"""

import scot.backend.sklearn     # use builtin (default) backend
import scot

from scot import plotting as splt
import matplotlib.pyplot as plt
from scot.datatools import randomize_phase

from scot.connectivity_statistics import *

import numpy as np

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


# Prepare the data
#
# Here we cut segments from 3s to 4s following each trigger out of the EEG. This
# is right in the middle of the motor imagery period.
data = scot.datatools.cut_segments(raweeg, triggers, 3 * fs, 4 * fs)


# Set up the analysis object
#
# We choose a VAR model order of 35, and reduction to 4 components.
ws = scot.Workspace({'model_order': 35}, reducedim=4, fs=fs, locations=locs)


fig = None


# Perform MVARICA and plot the components
ws.set_data(data, classes)
ws.do_mvarica(varfit='class')

p = ws.var_.test_whiteness(50)
print('Whiteness:', p)

fig = ws.plot_connectivity_topos(fig=fig)

p, s, _ = ws.compare_conditions(['hand'], ['foot'], 'ffDTF', repeats=100, plot=fig)

print(p)

# brep = 10
#
# print('foot')
# ws.set_data(data[:,:,classes=='foot'])
# b = ws.get_bootstrap_connectivity('ffDTF', brep)
# #ws.fit_var()
# #fig = ws.plot_connectivity('ffDTF', freq_range=[0, 30], fig=fig)
# fig = ws.plot_connectivity_bootstrap('ffDTF', freq_range=[0, 30], repeats=brep, fig=fig)
#
# print('hand')
# ws.set_data(data[:,:,classes=='hand'])
# a = ws.get_bootstrap_connectivity('ffDTF', brep)
# #ws.fit_var()
# #fig = ws.plot_connectivity('ffDTF', freq_range=[0, 30], fig=fig)
# fig = ws.plot_connectivity_bootstrap('ffDTF', freq_range=[0, 30], repeats=brep, fig=fig)
#
# print('diff')
#
# p = test_bootstrap_difference(a, b)
# for i in range(4):
#     p[i,i,:] = np.nan
# s = significance_fdr(p, 0.01)
# #fig = splt.plot_connectivity_spectrum(s*50, fs, freq_range=[0, 30], diagonal=-1, border=True, fig=fig)
#
# splt.plot_connectivity_significance(s, fs, freq_range=[0, 30], diagonal=-1, border=True, fig=fig)

ws.show_plots()
