# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2016 SCoT Development Team

from . import config


class BackendManager:
    def __init__(self):
        self.backends = {}
        self.current = None

    def register(self, name, activation_function):
        if config.getboolean('scot', 'verbose'):
            print('Registering backend:', name)
        self.backends[name] = activation_function

    def activate(self, name):
        if config.getboolean('scot', 'verbose'):
            print('Activating backend:', name)
        self.current = self.backends[name]()

    def names(self):
        return self.backends.keys()

    def items(self):
        return self.backends.items()

    def get_backend(self, name):
        return self.backends[name]()

    def __getitem__(self, item):
        return self.current[item]

    def __call__(self, name):
        self.activate(name)
