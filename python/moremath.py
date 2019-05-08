#!/usr/bin/env python

import functools

import autograd.numpy as np
import scipy.stats as stats
import uncertainties

def weightedaverage(values):
  values = tuple(values)
  if not values: raise IOError("Can't take the weighted average of an empty array")
  if all(x.std_dev == 0 for x in values):
    return sum(values) / len(values)
  return uncertainties.ufloat(
    sum(x.nominal_value / x.std_dev**2 for x in values) / sum(1 / x.std_dev**2 for x in values),
    sum(1 / x.std_dev**2 for x in values) ** -0.5
  )

def kspoissongaussian(mean, size=10000):
  np.random.seed(123456)  #make it deterministic
  return stats.ks_2samp(
    np.random.poisson(mean, size=size),
    np.random.normal(mean, mean**.5, size=size)
  ).statistic


def notnan(function):
  @functools.wraps(function)
  def newfunction(*args, **kwargs):
    result = function(*args, **kwargs)
    if np.any(np.isnan(result)):
      raise ValueError(
        "Calling {.__name__}({}): result = {}".format(
          function, " ".join(
            tuple(str(_) for _ in args) + tuple("{}={}".format(k, v) for k, v in kwargs.iteritems()),
          ), result
        )
      )
    return result
  return newfunction
