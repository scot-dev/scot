***************
Acknowledgments
***************

Overview
========

SCoT has been developed to provide connectivity estimation between brain sources for the Python community. SCoT was mainly developed by Martin Billinger as part of his PhD thesis at the `Institute for Knowledge Discovery`_, `Graz University of Technology`_, Graz, Austria. Although all source code was written from scratch, there are several existing sources that contributed directly or indirectly to this toolbox. Furthermore, there are a number of related connectivity toolboxes available for non-Python programming languages. In the following paragraphs, we would like to acknowledge these sources.

Related toolboxes
=================

SIFT_, the Source Information Flow Toolbox, is a connectivity toolbox for MATLAB. It has close ties with EEGLAB_, a widely used toolbox for EEG signal processing. Although SCoT has been developed independently, we would like to mention the excellent SIFT user manual (available on the SIFT website), which provides a very nice overview of connectivity estimation. Parts of SCoT, for instance the methods available for statistical significance tests, were inspired by the SIFT manual.

SCoT optionally uses scikit-learn_, a great machine learning package for Python, for some of its backend functionality. SCoT supports `Infomax ICA`_ from MNE_.

BioSig_ is also a MATLAB-based toolbox for biosignal processing. Besides providing I/O functionality for many biosignal file formats, it supports various signal processing and machine learning routines. It also provides functions to fit vector autoregressive models and derive various connectivity measures from the model parameters.

Relevant literature
===================

If you use SCoT, please consider citing the following reference publication:

    Martin Billinger, Clemens Brunner, Gernot R. Müller-Putz. SCoT: a Python toolbox for EEG source connectivity. Frontiers in Neuroinformatics, 8:22, 2014. `doi:10.3389/fninf.2014.00022`_

`Schlögl and Supp (2006)`_ provide an excellent overview of the various connectivity measures used in SCoT. The :ref:`api_reference` provides references to the original publication for each connectivity measure.

The MVARICA approach implemented in SCoT was developed by `Gómez-Herrero et al. (2008)`_.

.. _`Institute for Knowledge Discovery`: http://bci.tugraz.at/
.. _`Graz University of Technology`: http://www.tugraz.at/
.. _SIFT: http://sccn.ucsd.edu/wiki/SIFT
.. _EEGLAB: http://sccn.ucsd.edu/eeglab/
.. _BioSig: http://biosig.sourceforge.net/
.. _scikit-learn: http://scikit-learn.org/stable/
.. _`Infomax ICA`: http://dx.doi.org/10.1162/neco.1995.7.6.1129
.. _MNE: http://martinos.org/mne/stable/index.html
.. _`doi:10.3389/fninf.2014.00022`: http://dx.doi.org/10.3389/fninf.2014.00022
.. _`Schlögl and Supp (2006)`: http://dx.doi.org/10.1016/S0079-6123(06)59009-0
.. _`Gómez-Herrero et al. (2008)`: http://dx.doi.org/10.1016/j.neuroimage.2008.07.032
