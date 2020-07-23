# encoding: utf-8
"""
@author:  weijiandeng
@contact: dengwj16@gmail.com
"""

from .personX import personX
from .personX_spgan import personX_spgan
from .target_validation import target_validation
from .target_training import target_training
from .target_test import target_test

__factory = {
    'personX': personX,
    'personX_spgan': personX_spgan,
    'target_validation': target_validation,
    'target_training': target_training,
    'target_test': target_test,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
