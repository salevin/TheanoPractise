__author__ = 'slevin'

import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
import matplotlib.cm as cm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print