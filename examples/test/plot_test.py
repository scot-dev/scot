"""
==========================
Testing automatic examples
==========================

This will produce a simple image.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

plt.plot(np.random.randn(1000))
plt.show()
