from .dataset import ShapeNet, ScanObjectNN, ModelNet40, ShapeNetPart, S3DIS
from .model import autoencoder, classify, partseg, augment

import sys

class _C: # class Config(_C): def default(self): self.key = value
    def __init__(self, argv=sys.argv[1:]):
        self.default()
        if '-h' in argv or '--help' in argv or 'help' in argv:
            cfg_dict = [f'{i}: {j}' for i,j in self.__dict__.items()]
            print('\n    '.join(['Default config is:'] + cfg_dict))
            print("Acceptable parameter format:\n    [key, key=value]")
            exit()
        args = {p.split('=')[0]:p.split('=')[1] for p in argv if '=' in p}
        togs = {p:None for p in argv if '=' not in p}
        for key, val in self.__dict__.items():
            if type(val) is not bool and key in args:
                if val is not None:
                    args[key] = type(val)(args[key])
                self.__dict__[key] = args[key]
            if type(val) is bool and key in togs:
                self.__dict__[key] = not val
        self.func()
    def default(self): pass
    def func(self): pass
