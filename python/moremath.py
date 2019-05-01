#!/usr/bin/env python

import collections
import functools
import itertools
import subprocess

import autograd
import autograd.numpy as np
import scipy.optimize as optimize
import scipy.stats as stats
import uncertainties

import hom4pswrapper

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

def weightedaverage(values):
  values = tuple(values)
  if not values: raise IOError("Can't take the weighted average of an empty array")
  if all(x.std_dev == 0 for x in values):
    return sum(values) / len(values)
  return uncertainties.ufloat(
    sum(x.nominal_value / x.std_dev**2 for x in values) / sum(1 / x.std_dev**2 for x in values),
    sum(1 / x.std_dev**2 for x in values) ** -0.5
  )

@notnan
def cubicformula(coeffs):
  a, b, c, d = coeffs

  if a==0: return quadraticformula(b, c, d)

  Delta0 = b**2 - 3*a*c
  Delta1 = 2*b**3 - 9*a*b*c + 27*a**2*d

  if Delta0 == Delta1 == 0: return np.array([-b / (3*a)])

  C = ((Delta1 + (1 if Delta1>0 else -1) * (Delta1**2 - 4*Delta0**3 + 0j)**0.5) / 2) ** (1./3)

  xi = 0.5 * (-1 + (3**0.5)*1j)

  return np.array([-1/(3.*a) * (b + xi**k*C + Delta0/(xi**k * C)) for k in range(3)])

@notnan
def minimizequartic(coeffs):
  a, b, c, d, e = coeffs
  if a < 0: return -float("inf")
  if a == 0: return minimizecubic(coeffs[1:])
  flatpoints = cubicformula(np.array([4*a, 3*b, 2*c, d]))
  result = float("inf")
  for x in flatpoints:
    if abs(np.imag(x)) > 1e-12: continue
    x = np.real(x)
    result = min(result, a*x**4 + b*x**3 + c*x**2 + d*x + e)
  if np.isnan(result): assert False, (coeffs, result)
  return np.real(result)

def getquartic4d(coeffs):
  def quartic4d(x):
    assert len(x) == 4
    xand1 = np.array([x[0], x[1], x[2], x[3], 1.])
    return sum(
      np.prod((coeff,) + xs)
      for coeff, xs in itertools.izip_longest(
        coeffs,
        itertools.combinations_with_replacement(xand1, 4),
      )
    )
  return quartic4d

def getquartic4dmonomials(coeffs):
  xand1 = "1wxyz"
  for coeff, xs in itertools.izip_longest(
    coeffs,
    itertools.combinations_with_replacement(xand1, 4),
  ):
    yield coeff, collections.Counter(xs)

def differentiatemonomial(coeffandxs, variable):
  coeff, xs = coeffandxs
  xs = collections.Counter(xs)
  coeff *= xs[variable]
  if coeff: xs[variable] -= 1
  return coeff, xs

def getquartic4dgradientstrings(coeffs):
  monomials = tuple(getquartic4dmonomials(coeffs))
  derivatives = [], [], [], []
  variablesandderivatives = zip("wxyz", derivatives)
  for coeffandxs in monomials:
    for variable, derivative in variablesandderivatives:
      coeff, xs = differentiatemonomial(coeffandxs, variable)
      if coeff: derivative.append("*".join(itertools.chain((repr(coeff),), xs.elements())))
  return [" + ".join(_) + ";" for _ in derivatives]

def findcriticalpointsquartic4d(coeffs, cmdline=hom4pswrapper.smallparalleltdegnopostcmdline(), verbose=False):
  p = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdin = "\n".join(["{"] + getquartic4dgradientstrings(coeffs) + ["}"])
  out, err = p.communicate(stdin)
  if "error" in err:
    raise RuntimeError("hom4ps printed an error message.\n\ninput:\n\n"+stdin+"\n\nstdout:\n\n"+out+"\n\nstderr:\n\n"+err)
  if verbose: print out
  for solution in out.split("\n\n"):
    if "This solution appears to be real" in solution:
      yield [float(_) for _ in solution.split("\n")[-1].split()[1:]]

def minimizequartic4d(coeffs, verbose=False, **kwargs):
  quartic = getquartic4d(coeffs)
  criticalpoints = findcriticalpointsquartic4d(coeffs, verbose=verbose, **kwargs)
  minimum = float("inf")
  for cp in criticalpoints:
    value = quartic(cp)
    if verbose: print cp, value
    minimum = min(minimum, value)
  return minimum

def kspoissongaussian(mean, size=10000):
  np.random.seed(123456)  #make it deterministic
  return stats.ks_2samp(
    np.random.poisson(mean, size=size),
    np.random.normal(mean, mean**.5, size=size)
  ).statistic

if __name__ == "__main__":
  for size in range(0, 10000, 100):
    if not size: continue
    a = [kspoissongaussian(5, size=size) for i in range(100)]
    print "{:5d} {:5.2f} {:5.2f}".format(size, np.mean(a), np.std(a))
