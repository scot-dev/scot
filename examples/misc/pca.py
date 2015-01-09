"""
This example demonstrates that it is possible to reconstruct sources even if we
include a PCA step in the process.
"""

from __future__ import print_function

import numpy as np

from scot.pca import pca
from scot.var import VAR


# Generate data from a VAR(1) process
model0 = VAR(1)
model0.coef = np.array([[0.3, -0.6], [0, -0.9]])
x = model0.simulate(10000).squeeze()

# Transform data with PCA
w, v = pca(x)
y = x.dot(w)

print('Covariance of x:\n', np.cov(x.squeeze().T))
print('\nCovariance of y:\n', np.cov(y.squeeze().T))

model1, model2 = VAR(1), VAR(1)

# Fit model1 to the original data
model1.fit(x)

# Fit model2 to the PCA transformed data
model2.fit(y)

# The coefficients estimated on x (2) are exactly equal to the back-transformed
# coefficients estimated on y (4)
print('\n(1) True VAR coefficients:\n', model0.coef)
print('\n(2) VAR coefficients estimated on x:\n', model1.coef)
print('\n(3) VAR coefficients estimated on y:\n', model2.coef)
print('\n(4) VAR coefficients estimated on y and transformed back:\n',
      w.dot(model2.coef).dot(w.T))
