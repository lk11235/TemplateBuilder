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
    xand1 = np.array([1., x[0], x[1], x[2], x[3]])
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
    ctr = collections.Counter(xs)
    yield coeff, ctr

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

def findcriticalpointsquartic4d(coeffs, cmdline=hom4pswrapper.smallparalleltdegcmdline(), verbose=False):
  p = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdin = "\n".join(["{"] + getquartic4dgradientstrings(coeffs) + ["}"])
  if verbose: print stdin
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
  array1 = np.array([float(_) for _ in """
    40.6275075    -21.49057519   104.3954194   -114.87875327
    54.11702091    84.08289743   -92.16740827   139.28348042
   -87.81432722   533.8566524   -541.00589321   278.1779211
   254.23279963  -204.78919396   796.65771267   -11.47571644
   110.81018432   -52.39079616    85.00716321  -107.23477928
    52.73174679  -243.67726713  -233.37680382   321.20785972
  -149.21904821   339.79748175  -525.74064239   138.51768305
  -581.11749643  -396.39697679   590.5945439    642.30109756
  -512.09760584  -722.04862732   854.13904343    11.42095789
    -9.63820961    -5.90540961   -12.16713154    70.79916885
  -258.44891482    78.94393653   313.78865763  -251.49310358
   242.9260895    -54.30021115   203.8856044   -100.55939822
    -6.99804862    64.64983763  -226.75090536   -81.04075435
  -131.12260504   254.68181399   -38.10786336    71.08459885
   -71.49892346   138.21669971  -509.03041553    63.24360205
   603.02531661   -43.54912768  -604.67268387 -2138.93975872
  1121.79010246   338.93359808  -105.25702692  2458.36086402
  -375.21867853   272.94945156
  """.split()])
  array2 = np.array([float(_) for _ in """
    40.6275075     21.49057519   104.3954194   -114.87875327
    54.11702091    84.08289743    92.16740827  -139.28348042
    87.81432722   533.8566524   -541.00589321   278.1779211
   254.23279963  -204.78919396   796.65771267    11.47571644
   110.81018432   -52.39079616    85.00716321   107.23477928
   -52.73174679   243.67726713   233.37680382  -321.20785972
   149.21904821   339.79748175  -525.74064239   138.51768305
  -581.11749643  -396.39697679   590.5945439    642.30109756
  -512.09760584  -722.04862732   854.13904343    11.42095789
     9.63820961     5.90540961    12.16713154    70.79916885
  -258.44891482    78.94393653   313.78865763  -251.49310358
   242.9260895     54.30021115  -203.8856044    100.55939822
     6.99804862   -64.64983763   226.75090536    81.04075435
   131.12260504  -254.68181399    38.10786336    71.08459885
   -71.49892346   138.21669971  -509.03041553    63.24360205
   603.02531661   -43.54912768  -604.67268387 -2138.93975872
  1121.79010246   338.93359808  -105.25702692  2458.36086402
  -375.21867853   272.94945156
  """.split()])

  import argparse
  p = argparse.ArgumentParser()
  g = p.add_mutually_exclusive_group(required=True)
  g.add_argument("-a", action="store_true")
  g.add_argument("-b", action="store_true")
  g.add_argument("-c", action="store_true")
  args = p.parse_args()

  if args.a:
    minimizequartic4d(array1, verbose=True, cmdline=hom4pswrapper.smallparallelcmdline())
    raw_input()
    minimizequartic4d(array2, verbose=True, cmdline=hom4pswrapper.smallparallelcmdline())

  if args.b:
    minimizequartic4d(array1, verbose=True, cmdline=hom4pswrapper.smallparalleltdegcmdline())
    raw_input()
    minimizequartic4d(array2, verbose=True, cmdline=hom4pswrapper.smallparalleltdegcmdline())

  if args.c:
    minimizequartic4d(array1, verbose=True, cmdline=hom4pswrapper.smallparalleltdegnopostcmdline())
    raw_input()
    minimizequartic4d(array2, verbose=True, cmdline=hom4pswrapper.smallparalleltdegnopostcmdline())
