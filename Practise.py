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
    x = [1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83]
    y = [52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46]

    # TODO http://underflow.fr/ai/lets-play-with-theano-547  https://en.wikipedia.org/wiki/Simple_linear_regression
