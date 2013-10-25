# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

from __future__ import print_function

from uuid import uuid4
import numpy as np
import os, sys

binica_binary = os.path.dirname(os.path.relpath(__file__)) + '/binica/ica_linux'

#noinspection PyNoneFunctionAssignment,PyTypeChecker
def binica( data, binary = binica_binary ):
    """
    binica( data )

    Simple wrapper for the binica program.

    BINICA is bundled with EEGLAB, or can be downloaded from here:
        http://sccn.ucsd.edu/eeglab/binica/

    Parameters     Default  Shape   Description
    --------------------------------------------------------------------------
    data                  : n,m   : 2d data matrix (n samples, m signals)
    binary         : *    :       : path to the binica binary

    Output
    --------------------------------------------------------------------------
    w   weights matrix
    S   sphering matrix

    The unmixing matrix is obtained by multiplying U = np.dot(S,w)

    * by default the binary is expected to be "binica/ica_linux" relative
      to the directory where this module lies (typically scot/binica/ica_linux)
    """
    
    check_binary(binary)
    
    data = np.array( data, dtype=np.float32 )
    
    nframes, nchans = data.shape
    
    uid = uuid4()

    scriptfile = 'binica-%s.sc'%uid
    datafile = 'binica-%s.fdt'%uid
    weightsfile = 'binica-%s.wts'%uid
    #weightstmpfile = 'binicatmp-%s.wts'%uid
    spherefile = 'binica-%s.sph'%uid
    
    config = {'DataFile': datafile,
              'WeightsOutFile': weightsfile,
              'SphereFile': spherefile,
              'chans': nchans,
              'frames': nframes,
              'extended': 1}
    #    config['WeightsTempFile'] = weightstmpfile

    # create data file
    f = open( datafile, 'wb' )
    data.tofile(f)
    f.close()

    # create script file        
    f = open( scriptfile, 'wt' )
    for h in config:
        print(h, config[h], file=f)
    f.close()
    
    # flush output streams otherwise things printed before might appear after the ICA output.
    sys.stdout.flush()
    sys.stderr.flush()
    
    # run ICA    
    os.system(binary + ' < ' + scriptfile)
    
    os.remove(scriptfile)
    os.remove(datafile)    
    
    # read weights
    f = open( weightsfile, 'rb' )
    weights = np.fromfile( f, dtype=np.float32 )
    f.close()
    weights = np.reshape(weights, (nchans,nchans))
    
#    os.remove(weightstmpfile)
    os.remove(weightsfile)
    
    # read sphering matrix
    f = open( spherefile, 'rb' )
    sphere = np.fromfile( f, dtype=np.float32 )
    f.close()
    sphere = np.reshape(sphere, (nchans,nchans))    
    
    os.remove(spherefile)
    
    return weights, sphere
    

def check_binary(binary):
    """check if binary is available, and try to obtain it if not"""
    
    if os.path.exists(binary):
        return

    url = 'http://sccn.ucsd.edu/eeglab/binica/binica.zip'
    print(binary+' not found. Trying to download from '+url)

    path = os.path.dirname(binary)
        
    if not os.path.exists(path):
        os.makedirs(path)
    
    import urllib.request
    import zipfile
    import stat

    urllib.request.urlretrieve(url, path+'/binica.zip')
    
    with zipfile.ZipFile(path+'/binica.zip') as tgz:
        tgz.extractall(path+'/..')
    
    os.chmod(binary, stat.S_IXUSR)
