

***********************
Testing SPHINX Features
***********************

Syntax Highlighting
===================

.. sourcecode:: ipython

    In [69]: lines = plot([1,2,3])

    In [70]: setp(lines)
      alpha: float
      ...snip

Math
====

.. math::
  x_n = \sum_{i=1}^{p}a_ix_{n-i} + e_n

Plots
=====

.. plot:: pyplots/ellipses.py
   :include-source:


.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   x = np.random.randn(1000)
   plt.hist(x, 20)
   plt.grid()
   plt.title(r'Normal: $\mu=%.2f, \sigma=%.2f$'%(x.mean(), x.std()))
   plt.show()
