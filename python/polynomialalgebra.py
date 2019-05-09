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
    linearconstraint=np.array([1]),
  )

def minimizelinear(coeffs):
  """
  minimize y=a+bx
  """
  a, b = coeffs
  if not b:
    result = minimizeconstant(coeffs[:-1])
    x = result.x[0]
    result.linearconstraint = np.array([1, x])
    return result
  x = linearformula((a+2e6, b))[0]
  fun = a + b*x
  assert fun < -1e6
  return optimize.OptimizeResult(
    x=np.array([x]),
    success=False,
    status=3,
    message="function is linear, no minimum",
    fun=fun,
    linearconstraint=np.array([1, x]),
  )

def minimizequadratic(coeffs):
  """
  minimize y=a+bx+c
  """
  a, b, c = coeffs
  if c == 0:
    result = minimizelinear(coeffs[:-1])
    x = result.x[0]
    result.linearconstraint = np.array([1, x, x**2])
    return result
  if c < 0:
    x = quadraticformula((a+max(2e6, -a+2e6), b, c))[0]
    assert np.imag(x) == 0, x
    x = np.real(x)
    fun = a + b*x + c*x**2
    assert fun < -1e6
    return optimize.OptimizeResult(
      x=np.array([x]),
      success=False,
      status=3,
      message="function is negative quadratic, no minimum",
      fun=fun,
      linearconstraint=np.array([1, x, x**2]),
    )

  x = linearformula([b, 2*c])
  fun = a + b*x + c*x**2

  return optimize.OptimizeResult(
    x=np.array([x]),
    success=True,
    status=1,
    message="function is quadratic",
    fun=fun,
    linearconstraint=np.array([1, x, x**2]),
  )

def minimizecubic(coeffs):
  """
  minimize y=a+bx+cx^2+dx^3
  """
  a, b, c, d = coeffs
  if d == 0:
    result = minimizequadratic(coeffs[:-1])
    x = result.x[0]
    result.linearconstraint = np.array([1, x, x**2, x**3])
    return result
  x = [_ for _ in cubicformula((a+2e6, b, c, d)) if abs(np.imag(_)) < 1e-12][0]
  x = np.real(x)
  fun = a + b*x + c*x**2 + d*x**3
  assert fun < -1e6
  return optimize.OptimizeResult(
    x=np.array([x]),
    success=False,
    status=3,
    message="function is cubic, no minimum",
    fun=fun,
    linearconstraint=np.array([1, x, x**2, x**3])
  )

def minimizequartic(coeffs):
  """
  minimize y=a+bx+cx^2+dx^3+ex^4
  """
  a, b, c, d, e = coeffs
  if e == 0:
    result = minimizecubic(coeffs[:-1])
    x = result.x[0]
    result.linearconstraint = np.array([1, x, x**2, x**3, x**4])
    return result
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
      linearconstraint = np.array([1, x, x**2, x**3, x**4]),
    )

  flatpoints = cubicformula(np.array([b, 2*c, 3*d, 4*e]))
  minimum = float("inf")
  x = None
  for flatpoint in flatpoints:
    if abs(np.imag(flatpoint)) > 1e-12: continue
    flatpoint = np.real(flatpoint)
    newminimum = a + b*flatpoint + c*flatpoint**2 + d*flatpoint**3 + e*flatpoint**4
    if newminimum < minimum:
      minimum = newminimum
      x = flatpoint
  if np.isnan(minimum) or x is None: assert False, (coeffs, x, minimum)

  return optimize.OptimizeResult(
    x=np.array([x]),
    success=True,
    status=1,
    message="function is quartic",
    fun=np.real(minimum),
    linearconstraint = np.array([1, x, x**2, x**3, x**4]),
  )



def getpolynomialnd(d, n, coeffs, mirrorindices=()):
  mirrorarray = np.array([-1 if i in mirrorindices else 1 for i in xrange(n)])
  def polynomialnd(x):
    assert len(x) == n
    xand1 = np.concatenate(([1.], mirrorarray*x))
    return sum(
      np.prod((coeff,) + xs)
      for coeff, xs in itertools.izip_longest(
        coeffs,
        itertools.combinations_with_replacement(xand1, d),
      )
    )
  polynomialnd.__name__ = "polynomial{}degree{}d".format(d, n)
  return polynomialnd

def getnvariableletters(n):
  return "abcdefghijklmnopqrstuvwxyz"[-n:]

def getpolynomialndmonomials(d, n, coeffs, mirrorindices=()):
  xand1 = "1" + getnvariableletters(n)
  assert len(xand1) == n+1
  mirrorarray = np.concatenate(([1.], [-1 if i in mirrorindices else 1 for i in xrange(n)]))

  for coeff, xsandmirrors in itertools.izip_longest(
    coeffs,
    itertools.combinations_with_replacement(itertools.izip_longest(xand1, mirrorarray), d),
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

def getpolynomialndgradientstrings(d, n, coeffs):
  monomials = tuple(getpolynomialndmonomials(d, n, coeffs))
  derivatives = [[] for _ in xrange(n)]
  variablesandderivatives = zip(getnvariableletters(n), derivatives)
  for coeffandxs in monomials:
    for variable, derivative in variablesandderivatives:
      coeff, xs = differentiatemonomial(coeffandxs, variable)
      if coeff: derivative.append("*".join(itertools.chain((repr(coeff),), xs.elements())))
  return [" + ".join(_) + ";" for _ in derivatives]

def findcriticalpointspolynomialnd(d, n, coeffs, cmdline=hom4pswrapper.smallparalleltdegcmdline(), verbose=False):
  p = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdin = "\n".join(["{"] + getpolynomialndgradientstrings(d, n, coeffs) + ["}"])
  if verbose: print stdin
  out, err = p.communicate(stdin)
  if "error" in err:
    raise RuntimeError("hom4ps printed an error message.\n\ninput:\n\n"+stdin+"\n\nstdout:\n\n"+out+"\n\nstderr:\n\n"+err)
  if verbose: print out
  for solution in out.split("\n\n"):
    if "This solution appears to be real" in solution:
      yield [float(_) for _ in solution.split("\n")[-1].split()[1:]]

def minimizepolynomialnd(d, n, coeffs, verbose=False, **kwargs):
  polynomial = getpolynomialnd(d, n, coeffs)
  criticalpoints = list(findcriticalpointspolynomialnd(d, n, coeffs, verbose=verbose, **kwargs))
  minimum = float("inf")
  minimumx = None
  for cp in criticalpoints:
    value = polynomial(cp)
    if verbose: print cp, value
    if value < minimum:
      minimum = value
      minimumx = cp
  return optimize.OptimizeResult(
    x=np.array(minimumx),
    success=True,
    status=1,
    message="gradient is zero at {} real points".format(len(criticalpoints)),
    fun=minimum,
    linearconstraint=getpolynomialnd(d, n, np.diag([1 for _ in coeffs]))(minimumx)
  )

if __name__ == "__main__":
  a = [1, 1, -5, 0, 1]
  print minimizepolynomialnd(4, 1, a, verbose=True)
  print
  print minimizequartic(a)
