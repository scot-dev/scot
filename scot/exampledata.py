# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2015 SCoT Development Team

def load(dataset, datadir):
    """ Loads example dataset.
    
    If the requested dataset is not found in the location specified by
    `datadir` the function attempts to download it from the internet.
    
    Parameters
    ----------
    dataset : str
        Which dataset to load. Currently only 'motor imagery' is supported.
    datadir : str
        Path to the storage location of example datasets. Datasets are downloaded
        to this location if they cannot be found. If the directory does not exist
        it is created.
    Returns
    -------
        TODO: should we return a class (access elements by e.g. data.eeg, data.triggers, ...),
        a dict (access elements by e.g. data['eeg'], data['triggers'], ...), or simply a tuple?
    """
    
    TODO: implement
    
    
class Dataset:
      """ Base class for dataset definition. """
      def load():
          TODO: default loading mechanism
      def download():
          TODO: default downloader
      
      
class MotorImagery(Dataset):
    """ Motor Imagery example dataset. """
    files = ['motorimagery.mat']  # used by default Dataset.load()
    url = 'https://github.com/scot-dev/scot-data/raw/master/scotdata/'  # used by Dataset.download()
