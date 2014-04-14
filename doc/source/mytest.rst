

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


IPython
=======

.. ipython::

   In [136]: x = 2

   In [137]: x**3

Blablablab blab bldrnuaie blubb.

.. ipython::

   In [3]: z = x*3

   In [4]: z

   In [5]: print(z)

   In [6]: y = z()

And some plotting?

.. ipython::
   :suppress:

   In [1]: from pylab import *

.. ipython::

   In [1]: data = np.random.randn(10000)

   @savefig plot_simple.png width=10cm
   In [2]: plot(data);

   In [3]: figure();

   @savefig hist_simple.png width=10cm
   In [4]: hist(data, 100);
