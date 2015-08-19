# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013-2015 SCoT Development Team


from . import config


class BackendManager:
    def __init__(self):
        self.backends = {}
        self.current = None

    def register(self, name, activationfunction):
        if config.getboolean('scot', 'verbose'):
            print('Registering scot backend:', name)
        self.backends[name] = activationfunction

    def activate(self, name):
        if config.getboolean('scot', 'verbose'):
            print('Activating scot backend:', name)
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
