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

import numpy as np

import hom4pswrapper
from moremath import notnan
from optimizeresult import OptimizeResult

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
  return OptimizeResult(
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
    result.linearconstraint = np.array([0, x])
    return result
  x = linearformula((a+2e6, b))[0]
  fun = a + b*x
  assert fun < -1e6
  return OptimizeResult(
    x=np.array([x]),
    success=False,
    status=3,
    message="function is linear, no minimum",
    fun=fun,
    linearconstraint=np.array([0, x]),
  )

def minimizequadratic(coeffs):
  """
  minimize y=a+bx+c
  """
  a, b, c = coeffs
  if c == 0:
    result = minimizelinear(coeffs[:-1])
    x = result.x[0]
    result.linearconstraint = np.array([0, 0, 1])
    return result
  if c < 0:
    x = quadraticformula((a+max(2e6, -a+2e6), b, c))[0]
    assert np.imag(x) == 0, x
    x = np.real(x)
    fun = a + b*x + c*x**2
    assert fun < -1e6
    return OptimizeResult(
      x=np.array([x]),
      success=False,
      status=3,
      message="function is negative quadratic, no minimum",
      fun=fun,
      linearconstraint=np.array([0, 0, 1]),
    )

  x = linearformula([b, 2*c])[0]
  fun = a + b*x + c*x**2

  return OptimizeResult(
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
    result.linearconstraint = np.concatenate((result.linearconstraint, [0]))
    return result
  x = [_ for _ in cubicformula((a+2e6, b, c, d)) if abs(np.imag(_)) < 1e-12][0]
  x = np.real(x)
  fun = a + b*x + c*x**2 + d*x**3
  assert fun < -1e6
  return OptimizeResult(
    x=np.array([x]),
    success=False,
    status=3,
    message="function is cubic, no minimum",
    fun=fun,
    linearconstraint=np.array([0, 0, 0, x**3])
  )

def minimizequartic(coeffs):
  """
  minimize y=a+bx+cx^2+dx^3+ex^4
  """
  a, b, c, d, e = coeffs
  if e == 0:
    result = minimizecubic(coeffs[:-1])
    x = result.x[0]
    if result.linearconstraint[-1]:
      result.linearconstraint = np.array([0, 0, 0, 0, 1])
    else:
      result.linearconstraint = np.concatenate((result.linearconstraint, [0]))
    return result
  if e < 0:
    x = 1
    fun = 0
    while fun > -1e6:
      x *= 10
      fun = a + b*x + c*x**2 + d*x**3 + e*x**4
    return OptimizeResult(
      x=np.array([x]),
      success=False,
      status=3,
      message="function is negative quartic, no minimum",
      fun=fun,
      linearconstraint = np.array([0, 0, 0, 0, 1]),
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

  return OptimizeResult(
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

def getpolynomialndmonomials(d, n, coeffs=None, mirrorindices=()):
  xand1 = "1" + getnvariableletters(n)
  assert len(xand1) == n+1
  mirrorarray = np.concatenate(([1.], [-1 if i in mirrorindices else 1 for i in xrange(n)]))

  for coeff, xsandmirrors in itertools.izip_longest(
    coeffs if coeffs is not None else (),
    itertools.combinations_with_replacement(itertools.izip_longest(xand1, mirrorarray), d),
  ):
    if (coeffs is not coeff is None) or xsandmirrors is None:
      raise RuntimeError("Provided {} coefficients, need {}".format(len(coeffs), len(list(itertools.combinations_with_replacement(xand1, 4)))))
    xs, mirrors = itertools.izip(*xsandmirrors)
    ctr = collections.Counter(xs)
    if coeffs is None:
      yield ctr
    else:
      yield coeff * np.prod(mirrors), ctr

def getboundarymonomials(d, n, coeffs):
  firstletter = getnvariableletters(n)[0]
  for coeff, ctr in getpolynomialndmonomials(d, n, coeffs):
    if ctr["1"]: continue
    yield coeff, ctr

def differentiatemonomial(coeffandxs, variable):
  coeff, xs = coeffandxs
  xs = collections.Counter(xs)
  coeff *= xs[variable]
  if coeff: xs[variable] -= 1
  return coeff, xs

def getpolynomialndgradient(d, n, coeffs):
  monomials = tuple(getpolynomialndmonomials(d, n, coeffs))
  derivatives = [[] for _ in xrange(n)]
  variablesandderivatives = zip(getnvariableletters(n), derivatives)
  for coeffandxs in monomials:
    for variable, derivative in variablesandderivatives:
      coeff, xs = differentiatemonomial(coeffandxs, variable)
      if coeff: derivative.append((coeff, xs))
  return derivatives

def getpolynomialndgradientstrings(d, n, coeffs):
  return [
    " + ".join(
      "*".join(
        itertools.chain(
          (repr(coeff),), xs.elements()
        )
      ) for coeff, xs in derivative
    ) + ";" for derivative in getpolynomialndgradient(d, n, coeffs)
  ]

def findcriticalpointsquadraticnd(n, coeffs):
  gradient = getpolynomialndgradient(2, n, coeffs)
  variableletters = getnvariableletters(n)
  #Ax=b
  A = np.zeros((n, n))
  b = np.zeros((n, 1))
  for derivative, row, constant in itertools.izip_longest(gradient, A, b):
    for coeff, xs in derivative:
      xs = list(xs.elements())
      assert len(xs) <= 1
      if not xs or xs[0] == "1":
        constant[0] = -coeff  #note - sign here!  Because the gradient should be Ax + (-b)
      else:
        row[variableletters.index(xs[0])] = coeff
  return np.linalg.solve(A, b).T

def findcriticalpointspolynomialnd(d, n, coeffs, verbose=False, usespecialcases=True):
  if usespecialcases and d == 2:
    for _ in findcriticalpointsquadraticnd(n, coeffs): yield _
    return
  stdin = "\n".join(["{"] + getpolynomialndgradientstrings(d, n, coeffs) + ["}"])
  try:
    out = hom4pswrapper.runhom4ps(stdin, whichcmdline="smallparalleltdeg", verbose=verbose)
  except hom4pswrapper.Hom4PSFailedPathsError as e1:
    try:
      out = hom4pswrapper.runhom4ps(stdin, whichcmdline="smallparallel", verbose=verbose)
    except hom4pswrapper.Hom4PSFailedPathsError as e2:
      try:
        out = hom4pswrapper.runhom4ps(stdin, whichcmdline="easy", verbose=verbose)
      except hom4pswrapper.Hom4PSFailedPathsError as e3:
        out = min(e1, e2, e3, key=lambda x: x.nfailedpaths).stdout
  for solution in out.split("\n\n"):
    if "This solution appears to be real" in solution:
      yield [float(_) for _ in solution.split("\n")[-1].split()[1:]]

class NoCriticalPointsError(ValueError):
  def __init__(self, coeffs):
    super(NoCriticalPointsError, self).__init__("can't find critical points for polynomial: {}".format(coeffs))

def printresult(function):
  def newfunction(*args, **kwargs):
    result = function(*args, **kwargs)
    print args, kwargs
    print result
    raw_input()
    return result
  return newfunction

#@printresult
def minimizepolynomialnd(d, n, coeffs, verbose=False, **kwargs):
  if n == 1:
    if d == 0: return minimizeconstant(coeffs)
    if d == 1: return minimizelinear(coeffs)
    if d == 2: return minimizequadratic(coeffs)
    if d == 3: return minimizecubic(coeffs)
    if d == 4: return minimizequartic(coeffs)

  if not np.any(coeffs[1:]):
    return OptimizeResult(
      x=np.array([0]*n),
      success=True,
      status=2,
      message="polynomial is constant",
      fun=coeffs[0],
      linearconstraint=np.array([1 if i==0 else 0 for i in range(len(coeffs))])
    )

  polynomial = getpolynomialnd(d, n, coeffs)

  #check the behavior around the sphere at infinity
  boundarycoeffs, boundarymonomials = zip(*getboundarymonomials(d, n, coeffs))
  boundarycoeffs = np.array(boundarycoeffs)
  boundaryresult = minimizepolynomialnd(d, n-1, boundarycoeffs, verbose=verbose, **kwargs)
  if boundaryresult.fun < 0:
    x = np.concatenate(([1], boundaryresult.x))
    multiply = 1
    while polynomial(x*multiply) > -1e6:
      multiply *= 10
      if multiply > 1e30: assert False

    linearconstraint = []
    monomials = getpolynomialndmonomials(d, n)
    boundarylinearconstraint = iter(boundaryresult.linearconstraint)
    boundarymonomials = iter(boundarymonomials)
    nextboundarymonomial = next(boundarymonomials)

    for monomial in monomials:
      if monomial == nextboundarymonomial:
        linearconstraint.append(next(boundarylinearconstraint))
        nextboundarymonomial = next(boundarymonomials, None)
      else:
        linearconstraint.append(0)

    for remaining in itertools.izip_longest(boundarylinearconstraint, boundarymonomials): assert False
    assert nextboundarymonomial is None

    return OptimizeResult(
      x=x*multiply,
      success=False,
      status=3,
      message="function goes to -infinity somewhere around the sphere at infinity",
      fun=polynomial(x*multiply),
      linearconstraint=np.array(linearconstraint),
      boundaryresult=boundaryresult,
    )

  criticalpoints = list(findcriticalpointspolynomialnd(d, n, coeffs, verbose=verbose, **kwargs))
  minimum = float("inf")
  minimumx = None
  for cp in criticalpoints:
    value = polynomial(cp)
    if verbose: print cp, value
    if value < minimum:
      minimum = value
      minimumx = cp
  if minimumx is None:
    raise NoCriticalPointsError(coeffs)

  linearconstraint = getpolynomialnd(d, n, np.diag([1 for _ in coeffs]))(minimumx)
  if not np.isclose(np.dot(linearconstraint, coeffs), minimum):
    raise ValueError("{} != {}??".format(np.dot(linearconstraint, coeffs), minimum))

  return OptimizeResult(
    x=np.array(minimumx),
    success=True,
    status=1,
    message="gradient is zero at {} real points".format(len(criticalpoints)),
    fun=minimum,
    linearconstraint=linearconstraint,
    boundaryresult=boundaryresult,
  )

def coeffswithpermutedvariables(d, n, coeffs, permutationdict):
  xand1 = "1"+getnvariableletters(n)
  monomialsandcoeffs = {frozenset(ctr.iteritems()): coeff for coeff, ctr in getpolynomialndmonomials(d, n, coeffs)}
  for monomial in getpolynomialndmonomials(d, n):
    newmonomial = collections.Counter(permutationdict[e] for e in monomial.elements())
    newcoeff = monomialsandcoeffs[frozenset(newmonomial.iteritems())]
    yield newcoeff

def minimizepolynomialnd_permutation(d, n, coeffs, permutationdict, **kwargs):
  newcoeffs = np.array(list(coeffswithpermutedvariables(d, n, coeffs, permutationdict)))
  result = minimizepolynomialnd(d, n, newcoeffs, **kwargs)

  reverse = {v: k for k, v in permutationdict.iteritems()}

  if all(k == v for k, v in permutationdict.iteritems()): return result

  linearconstraint = np.array(list(coeffswithpermutedvariables(d, n, result.linearconstraint, reverse)))
  if permutationdict["1"] == "1" and not np.isclose(np.dot(linearconstraint, coeffs), result.fun):
    raise ValueError("{} != {}??".format(np.dot(linearconstraint, coeffs), result.fun))
  if np.sign(np.dot(linearconstraint, coeffs)) != np.sign(result.fun):
    raise ValueError("sign({}) != sign({})??".format(np.dot(linearconstraint, coeffs), result.fun))

  return OptimizeResult(
    permutation=permutationdict,
    permutedresult=result,
    fun=result.fun,
    linearconstraint=linearconstraint,
    x=(result.x, "permuted")
  )

def minimizepolynomialnd_permutations(d, n, coeffs, debugprint=False, **kwargs):
  xand1 = "1"+getnvariableletters(n)
  best = None
  for permutation in itertools.permutations(xand1):
    permutationdict = {orig: new for orig, new in itertools.izip(xand1, permutation)}
    try:
      result = minimizepolynomialnd_permutation(d, n, coeffs, permutationdict=permutationdict, **kwargs)
    except NoCriticalPointsError:
      continue

    #want this to be small
    nonzerolinearconstraint = result.linearconstraint[np.nonzero(result.linearconstraint)]
    figureofmerit = len(result.linearconstraint) - len(nonzerolinearconstraint), sum(np.log(abs(nonzerolinearconstraint))**2)

    if debugprint:
      print "---------------------------------"
      print result.linearconstraint
      print figureofmerit

    if best is None or figureofmerit < best[3]:
      best = permutation, permutationdict, result, figureofmerit
      if debugprint: print "new best"

    if debugprint: print "---------------------------------"

    if result.fun > 0 or figureofmerit <= (0, 50):
      break

  permutation, permutationdict, result, figureofmerit = best

  return result

if __name__ == "__main__":
  coeffs = np.array([
    7.14562045e-06, -5.77999470e-07,  8.02158736e-06,  1.19417131e-05,
    4.58641642e-06,  8.61578331e-07, -9.05128851e-07, -1.12497735e-06,
   -6.14150255e-07,  1.39521049e-06,  5.74903386e-06,  1.76204796e-06,
    5.17540756e-06,  2.32619519e-06,  1.94951576e-05
  ])

  print np.array(list(findcriticalpointspolynomialnd(2, 4, coeffs)))
  print np.array(list(findcriticalpointspolynomialnd(2, 4, coeffs, usespecialcases=False)))
  print findcriticalpointsquadraticnd(4, coeffs)
