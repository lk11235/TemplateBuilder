#!/usr/bin/env python

"""
All functions in this module take the coefficients in this order:
1, x, y, z, ..., x^2, xy, xz, ..., x^3, x^2y, x^2z, ......

linearformula, quadraticformula, and cubicformula solve a + bx (+ cx^2 (+dx^3)) = 0
and return the results as a numpy array of solutions.

The minimize functions all return an OptimizeResults.
If the minimum is actually -inf, the success will be False and the reported results
will be a number less than -1e6, with some set of xs that produce that result.
"""

from __future__ import division

import collections
import functools
import itertools
import subprocess

import autograd
import autograd.numpy as np
import scipy.optimize as optimize

import hom4pswrapper
from moremath import notnan

@notnan
def linearformula(coeffs):
  """
  solve a + bx = 0
  """
  a, b = coeffs
  if b==0: raise ValueError("function is a constant")
  return np.array([-a/b])

@notnan
def quadraticformula(coeffs):
  """
  solve a + bx + cx^2 = 0

  NOTE this is consistent with the convention here, but the opposite of the normal convention
  """
  a, b, c = coeffs
  if c == 0: return linearformula(a, b)
  return (-b + np.array([1, -1]) * (b**2 - 4*c*a + 0j)**0.5) / (2*c)

@notnan
def cubicformula(coeffs):
  """
  solve a + bx + cx^2 + dx^3 = 0

  NOTE this is consistent with the convention here, but the opposite of the normal convention (e.g. wikipedia)
  """
  a, b, c, d = coeffs

  if d==0: return quadraticformula(a, b, c)

  Delta0 = c**2 - 3*d*b
  Delta1 = 2*c**3 - 9*d*c*b + 27*d**2*a

  if Delta0 == Delta1 == 0: return np.array([-c / (3*d)])

  C = ((Delta1 + (1 if Delta1>0 else -1) * (Delta1**2 - 4*Delta0**3 + 0j)**0.5) / 2) ** (1./3)

  xi = 0.5 * (-1 + (3**0.5)*1j)

  return np.array([-1/(3.*d) * (c + xi**k*C + Delta0/(xi**k * C)) for k in range(3)])

def minimizeconstant(coeffs):
  """
  minimize y=a
  """
  a, = coeffs
  return optimize.OptimizeResult(
    x=np.array([0]),
    success=True,
    status=2,
    message="function is constant",
    fun=a,
  )

def minimizelinear(coeffs):
  """
  minimize y=a+bx
  """
  a, b = coeffs
  if not b: return minimizeconstant(coeffs[:-1])
  x = linearformula((a+2e6, b))[0]
  fun = a + b*x
  assert fun < -1e6
  return optimize.OptimizeResult(
    x=np.array([x]),
    success=False,
    status=3,
    message="function is linear, no minimum",
    fun=fun
  )

def minimizequadratic(coeffs):
  """
  minimize y=a+bx+c
  """
  a, b, c = coeffs
  if c == 0: return minimizelinear(coeffs[:-1])
  if c < 0:
    x = quadraticformula((a+max(2e6, -a+2e6), b, c))[0]
    assert x.imag == 0, x
    x = x.real
    fun = a + b*x + c*x**2
    assert fun < -1e6
    return optimize.OptimizeResult(
      x=np.array([x]),
      success=False,
      status=3,
      message="function is negative quadratic, no minimum",
      fun=fun
    )

  x = linearformula([b, 2*c])
  fun = a + b*x + c*x**2

  return optimize.OptimizeResult(
    x=np.array([x]),
    success=True,
    status=1,
    message="function is quadratic",
    fun=fun,
  )

def minimizecubic(coeffs):
  """
  minimize y=a+bx+cx^2+dx^3
  """
  a, b, c, d = coeffs
  if d == 0: return minimizequadratic(coeffs[:-1])
  x = [_ for _ in cubicformula((a+2e6, b, c, d)) if abs(_.imag) < 1e-12][0]
  x = x.real
  fun = a + b*x + c*x**2 + d*x**3
  assert fun < -1e6
  return optimize.OptimizeResult(
    x=np.array([x]),
    success=False,
    status=3,
    message="function is cubic, no minimum",
    fun=fun
  )

def minimizequartic(coeffs):
  """
  minimize y=a+bx+cx^2+dx^3+ex^4
  """
  a, b, c, d, e = coeffs
  if e == 0: return minimizecubic(coeffs[:-1])
  if e < 0:
    x = 1
    fun = 0
    while fun > -1e6:
      x *= 10
      fun = a + b*x + c*x**2 + d*x**3 + e*x**4
    return optimize.OptimizeResult(
      x=np.array([x]),
      success=False,
      status=3,
      message="function is negative quartic, no minimum",
      fun=fun,
    )

  flatpoints = cubicformula(np.array([b, 2*c, 3*d, 4*e]))
  result = float("inf")
  xresult = None
  for x in flatpoints:
    if abs(np.imag(x)) > 1e-12: continue
    x = np.real(x)
    newresult = a + b*x + c*x**2 + d*x**3 + e*x**4
    if newresult < result:
      result = newresult
      xresult = x
  if np.isnan(result) or xresult is None: assert False, (coeffs, xresult, result)

  return optimize.OptimizeResult(
    x=np.array([xresult]),
    success=True,
    status=1,
    message="function is quartic",
    fun=np.real(result),
  )

def getquarticnd(n, coeffs, mirrorindices=()):
  mirrorarray = np.array([-1 if i in mirrorindices else 1 for i in xrange(n)])
  def quarticnd(x):
    assert len(x) == n
    xand1 = np.concatenate(([1.], mirrorarray*x))
    return sum(
      np.prod((coeff,) + xs)
      for coeff, xs in itertools.izip_longest(
        coeffs,
        itertools.combinations_with_replacement(xand1, 4),
      )
    )
  quarticnd.__name__ = "quartic{}d".format(n)
  return quarticnd

def getnvariableletters(n):
  return "abcdefghijklmnopqrstuvwxyz"[-n:]

def getquarticndmonomials(n, coeffs, mirrorindices=()):
  xand1 = "1" + getnvariableletters(n)
  assert len(xand1) == n+1
  mirrorarray = np.concatenate(([1.], [-1 if i in mirrorindices else 1 for i in xrange(n)]))

  for coeff, xsandmirrors in itertools.izip_longest(
    coeffs,
    itertools.combinations_with_replacement(itertools.izip_longest(xand1, mirrorarray), 4),
  ):
    if coeff is None or xsandmirrors is None:
      raise RuntimeError("Provided {} coefficients, need {}".format(len(coeffs), len(list(itertools.combinations_with_replacement(xand1, 4)))))
    xs, mirrors = itertools.izip(*xsandmirrors)
    ctr = collections.Counter(xs)
    yield coeff * np.prod(mirrors), ctr

def differentiatemonomial(coeffandxs, variable):
  coeff, xs = coeffandxs
  xs = collections.Counter(xs)
  coeff *= xs[variable]
  if coeff: xs[variable] -= 1
  return coeff, xs

def getquarticndgradientstrings(n, coeffs):
  monomials = tuple(getquarticndmonomials(n, coeffs))
  derivatives = [[] for _ in xrange(n)]
  variablesandderivatives = zip(getnvariableletters(n), derivatives)
  for coeffandxs in monomials:
    for variable, derivative in variablesandderivatives:
      coeff, xs = differentiatemonomial(coeffandxs, variable)
      if coeff: derivative.append("*".join(itertools.chain((repr(coeff),), xs.elements())))
  return [" + ".join(_) + ";" for _ in derivatives]

def findcriticalpointsquarticnd(n, coeffs, cmdline=hom4pswrapper.smallparalleltdegcmdline(), verbose=False):
  p = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdin = "\n".join(["{"] + getquarticndgradientstrings(n, coeffs) + ["}"])
  if verbose: print stdin
  out, err = p.communicate(stdin)
  if "error" in err:
    raise RuntimeError("hom4ps printed an error message.\n\ninput:\n\n"+stdin+"\n\nstdout:\n\n"+out+"\n\nstderr:\n\n"+err)
  if verbose: print out
  for solution in out.split("\n\n"):
    if "This solution appears to be real" in solution:
      yield [float(_) for _ in solution.split("\n")[-1].split()[1:]]

def minimizequarticnd(n, coeffs, verbose=False, **kwargs):
  quartic = getquarticnd(n, coeffs)
  criticalpoints = findcriticalpointsquarticnd(n, coeffs, verbose=verbose, **kwargs)
  minimum = float("inf")
  minimumx = None
  for cp in criticalpoints:
    value = quartic(cp)
    if verbose: print cp, value
    if value < minimum:
      minimum = value
      minimumx = cp
  return minimum

if __name__ == "__main__":
  print minimizequarticnd(4, np.array(range(70)) * np.array([(-1)**x for x in range(70)]), verbose=True)
