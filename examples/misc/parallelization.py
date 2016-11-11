# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team

"""
This example shows how to parallelize certain computations in SCoT.
"""

import numpy as np
import time

from scot.datatools import cut_segments
from scot.var import VAR


# The data set contains a continuous 45 channel EEG recording of a motor
# imagery experiment. The data was preprocessed to reduce eye movement
# artifacts and resampled to a sampling rate of 100 Hz. With a visual cue, the
# subject was instructed to perform either hand or foot motor imagery. The
# trigger time points of the cues are stored in 'triggers', and 'classes'
# contains the class labels. Duration of the motor imagery period was
# approximately six seconds.
from scot.datasets import fetch


# Prevent execution of the main script in worker threads
if __name__ == "__main__":

    midata = fetch("mi")[0]

    raweeg = midata["eeg"]
    triggers = midata["triggers"]
    classes = midata["labels"]
    fs = midata["fs"]
    locs = midata["locations"]

    # Prepare data
    #
    # Here we cut out segments from 3s to 4s after each trigger. This is right
    # in the middle of the motor imagery period.
    data = cut_segments(raweeg, triggers, 3 * fs, 4 * fs)

    # only use every 10th trial to make the example run faster
    data = data[::10]

    var = VAR(model_order=5)
    var.fit(data)
    for n_jobs in [-1, None, 1, 2, 3, 4, 5, 6, 7, 8]:
        # Set random seed for repeatable results
        np.random.seed(42)
        var.n_jobs = n_jobs
        start = time.perf_counter()
        p = var.test_whiteness(10, repeats=1000)
        time1 = time.perf_counter()
        print('n_jobs: {:>4s}, whiteness test: {:.2f}s, p = {}'.format(str(n_jobs), time1 - start, p))
