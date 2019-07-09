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
from moremath import closebutnotequal, notnan
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
      raise RuntimeError("Provided {} coefficients, need {}".format(len(coeffs), len(list(itertools.combinations_with_replacement(xand1, d)))))
    xs, mirrors = itertools.izip(*xsandmirrors)
    ctr = collections.Counter(xs)
    if coeffs is None:
      yield ctr
    else:
      yield coeff * np.prod(mirrors), ctr

class DegeneratePolynomialError(ValueError):
  def __init__(self, coeffs, d, variable):
    super(DegeneratePolynomialError, self).__init__("Can't find the boundary polynomial of {} because it has 0 in front of the {}^{} term".format(coeffs, d, variable))

def getboundarymonomials(d, n, coeffs):
  firstletter = getnvariableletters(n)[0]
  for coeff, ctr in getpolynomialndmonomials(d, n, coeffs):
    if ctr["1"]: continue
    if len(set(ctr.elements())) == 1 and coeff == 0: raise DegeneratePolynomialError(coeffs, d, next(ctr.elements()))
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

def getpolynomialndgradientstrings(d, n, coeffs, homogenize=False):
  return [
    " + ".join(
      "*".join(
        itertools.chain(
          (repr(coeff),), ["alpha" if homogenize and variable=="1" else variable for variable in xs.elements()]
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

def findcriticalpointspolynomialnd(d, n, coeffs, verbose=False, usespecialcases=True, cmdlinestotry=("smallparalleltdeg",), homogenizecoeffs=None, boundarycriticalpoints=[], setsmallestcoefficientsto0=False):
  if usespecialcases and d == 2:
    return findcriticalpointsquadraticnd(n, coeffs)

  gradientstrings = getpolynomialndgradientstrings(d, n, coeffs, homogenize=homogenizecoeffs is not None)
  if homogenizecoeffs is not None:
    extraequations = [
      "+".join("{:g}".format(coeff)+"*"+variable for coeff, variable in itertools.izip_longest(homogenizecoeffs, itertools.chain(["alpha"], getnvariableletters(n), ["1"]))) + ";"
    ]
  else:
    extraequations = []
  stdin = "\n".join(["{"] + gradientstrings + extraequations + ["}"])

  errors = []
  for cmdline in cmdlinestotry:
    try:
      result = hom4pswrapper.runhom4ps(stdin, whichcmdline=cmdline, verbose=verbose)
    except hom4pswrapper.Hom4PSFailedPathsError as e:
      errors.append(e)
    except hom4pswrapper.Hom4PSDivergentPathsError as e:
      if homogenizecoeffs is None:
        for cp in boundarycriticalpoints:
          try:
            newhomogenizecoeffs = np.concatenate(([1, 1], 1/cp, [1]))
          except RuntimeWarning as runtimewarning:
            if "divide by zero encountered in true_divide" in runtimewarning:
              continue #can't use this cp
          try:
            homogenizedresult = findcriticalpointspolynomialnd(d, n, coeffs, verbose=verbose, usespecialcases=usespecialcases, cmdlinestotry=cmdlinestotry, homogenizecoeffs=newhomogenizecoeffs, setsmallestcoefficientsto0=setsmallestcoefficientsto0)
          except NoCriticalPointsError:
            pass
          else:
            for solution in e.realsolutions:
              if not any(np.allclose(solution, newsolution) for newsolution in homogenizedresult):
                break
            else: #all old solutions are still there after homogenizing
              return homogenizedresult
      errors.append(e)
    else:
      solutions = result.realsolutions
      if homogenizecoeffs is not None:
        solutions = [solution[1:] / solution[0] for solution in solutions]
      return solutions

  if verbose:
    print "seeing if those calls gave different solutions, in case between them we have them all covered"

  solutions = []
  allclosekwargs = {"rtol": 1e-3, "atol": 1e-08}
  for error in errors:
    thesesolutions = error.realsolutions

    while any(closebutnotequal(first, second, **allclosekwargs) for first, second in itertools.combinations(thesesolutions, 2)):
      allclosekwargs["rtol"] /= 2
      allclosekwargs["atol"] /= 2

    for newsolution in thesesolutions:
      if not any(closebutnotequal(newsolution, oldsolution, **allclosekwargs) for oldsolution in solutions):
        solutions.append(newsolution)

  numberofpossiblesolutions = min(len(e.solutions) + e.nfailedpaths + e.ndivergentpaths for e in errors)

  if len(solutions) > numberofpossiblesolutions:
    raise NoCriticalPointsError(coeffs, moremessage="found too many critical points in the union of the different configurations", solutions=solutions)

  if len(solutions) == numberofpossiblesolutions:
    if verbose: print "we do"
    if homogenizecoeffs is not None:
      solutions = [solution[1:] / solution[0] for solution in solutions]
    return solutions

  if setsmallestcoefficientsto0:
    newcoeffs = []
    setto0 = []
    biggest = max(abs(coeffs))
    smallest = min(abs(coeffs[np.nonzero(coeffs)]))
    for coeff in coeffs:
      if coeff == 0 or np.log(biggest / abs(coeff)) < np.log(abs(coeff) / smallest):
        newcoeffs.append(coeff)
      else:
        setto0.append(coeff)
        newcoeffs.append(0)
    newcoeffs = np.array(newcoeffs)
    setto0 = np.array(setto0)
    if np.log10(min(abs(newcoeffs[np.nonzero(newcoeffs)])) / max(abs(setto0))) > np.log10(biggest / smallest) / 3: #if there's a big gap
      if verbose: print "trying again after setting the smallest coefficients to 0:\n{}".format(setto0)
      newsolutions = findcriticalpointspolynomialnd(d, n, newcoeffs, verbose=verbose, usespecialcases=usespecialcases, cmdlinestotry=cmdlinestotry, homogenizecoeffs=homogenizecoeffs, boundarycriticalpoints=boundarycriticalpoints)
      for oldsolution in solutions:
        if verbose: print "checking if old solution {} is still here".format(oldsolution)
        if not any(np.allclose(oldsolution, newsolution, **allclosekwargs) for newsolution in newsolutions):
          if verbose: print "it's not"
          break  #removing this coefficient messed up one of the old solutions, so we can't trust the new ones
        if verbose: print "it is"
      else:  #removing this coefficient didn't mess up the old solutions
        return newsolutions
    else:
      if verbose: print "can't set the smallest coefficients to 0, there's not a clear separation between big and small:\nbig candidates:{} --> range = {} - {}\nsmall candidates: {} --> range = {} - {}\n\nmore info: {} {}".format(newcoeffs[np.nonzero(newcoeffs)], min(abs(newcoeffs[np.nonzero(newcoeffs)])), max(abs(newcoeffs)), setto0, min(abs(setto0)), max(abs(setto0)), np.log10(min(abs(newcoeffs[np.nonzero(newcoeffs)])) / max(abs(setto0))), np.log10(biggest / smallest))

  raise NoCriticalPointsError(coeffs, moremessage="there are failed and/or divergent paths, even after trying different configurations and saving mechanisms", solutions=solutions)

class NoCriticalPointsError(ValueError):
  def __init__(self, coeffs, moremessage=None, solutions=None):
    message = "error finding critical points for polynomial: {}".format(coeffs)
    if moremessage: message += "\n\n"+moremessage
    super(NoCriticalPointsError, self).__init__(message)
    self.coeffs = coeffs
    self.solutions = solutions

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
  boundarykwargs = kwargs.copy()
  if kwargs.get("homogenizecoeffs") is not None:
    boundarykwargs["homogenizecoeffs"] = kwargs["homogenizecoeffs"][1:]
  boundaryresult = minimizepolynomialnd(d, n-1, boundarycoeffs, verbose=verbose, **boundarykwargs)
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

  assert "boundarycriticalpoints" not in kwargs
  if hasattr(boundaryresult, "criticalpoints"):
    kwargs["boundarycriticalpoints"] = boundaryresult.criticalpoints

  criticalpoints = list(findcriticalpointspolynomialnd(d, n, coeffs, verbose=verbose, **kwargs))
  if not criticalpoints:
    raise NoCriticalPointsError(coeffs, moremessage="system of polynomials doesn't have any critical points")

  criticalpoints.sort(key=polynomial)
  if verbose:
    for cp in criticalpoints:
      print cp, polynomial(cp)
  minimumx = criticalpoints[0]
  minimum = polynomial(minimumx)

  linearconstraint = getpolynomialnd(d, n, np.diag([1 for _ in coeffs]))(minimumx)
  if not np.isclose(np.dot(linearconstraint, coeffs), minimum, rtol=2e-2):
    raise ValueError("{} != {}??".format(np.dot(linearconstraint, coeffs), minimum))

  return OptimizeResult(
    x=np.array(minimumx),
    success=True,
    status=1,
    message="gradient is zero at {} real points".format(len(criticalpoints)),
    fun=minimum,
    linearconstraint=linearconstraint,
    boundaryresult=boundaryresult,
    criticalpoints=criticalpoints,
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
  if (
    np.sign(np.dot(linearconstraint, coeffs)) != np.sign(result.fun)
    and np.sign(np.dot(linearconstraint, coeffs)) != 0
    and np.sign(result.fun) != 0
    and not np.isclose(np.dot(linearconstraint, coeffs), result.fun) #numerical issues occasionally give +epsilon and -epsilon when you add in different orders
  ):
    raise ValueError("sign({}) != sign({})??".format(np.dot(linearconstraint, coeffs), result.fun))

  return OptimizeResult(
    permutation=permutationdict,
    permutedresult=result,
    fun=result.fun,
    linearconstraint=linearconstraint,
    x=(result.x, "permuted")
  )

def permutations_differentonesfirst(iterable):
  """
  Variation of itertools.permutations:
    it tries to find the one that is most different from all the ones
    returned so far.  For example, permutations_finddifferentones("ABCD")
    starts by yielding ABCD as usual, but then goes to BADC
    followed by CDAB and DCBA
  """
  tpl = tuple(iterable)
  permutations = list(itertools.permutations(tpl))
  done = []
  while permutations:
    best = None
    bestfom = -float("inf")
    for permutation in permutations:
      figureofmerit = sum(sum(x != y for x, y in itertools.izip(permutation, oldpermutation)) for oldpermutation in done)
      if figureofmerit > bestfom:
        bestfom = figureofmerit
        best = permutation
    done.append(best)
    permutations.remove(best)
    yield best


def minimizepolynomialnd_permutations(d, n, coeffs, debugprint=False, permutationmode="best", **kwargs):
  xand1 = "1"+getnvariableletters(n)
  best = None
  signs = {1: [], -1: [], 0: []}
  for permutation in permutations_differentonesfirst(xand1):
    permutationdict = {orig: new for orig, new in itertools.izip(xand1, permutation)}
    try:
      result = minimizepolynomialnd_permutation(d, n, coeffs, permutationdict=permutationdict, **kwargs)
    except (NoCriticalPointsError, DegeneratePolynomialError):
      continue

    #want this to be small
    nonzerolinearconstraint = result.linearconstraint[np.nonzero(result.linearconstraint)]
    figureofmerit = (result.fun >= 0), len(result.linearconstraint) - len(nonzerolinearconstraint), sum(np.log(abs(nonzerolinearconstraint))**2)

    if debugprint:
      print "---------------------------------"
      print result.linearconstraint
      print figureofmerit

    if best is None or figureofmerit < best[3]:
      best = permutation, permutationdict, result, figureofmerit
      if debugprint: print "new best"

    if debugprint: print "---------------------------------"

    signs[np.sign(result.fun)].append(permutation)
    if {
      "best": result.fun > 0 or figureofmerit <= (False, 0, 50),
      "asneeded": True,
      "best_gothroughall": False,
    }[permutationmode]:
      break

  if best is None:
    if "setsmallestcoefficientsto0" not in kwargs:
      kwargs["setsmallestcoefficientsto0"] = True
      return minimizepolynomialnd_permutations(d, n, coeffs, debugprint=debugprint, permutationmode=permutationmode, **kwargs)
    if "cmdlinestotry" not in kwargs:
      kwargs["cmdlinestotry"] = "smallparalleltdeg", "smallparallel"#, "easy"
      return minimizepolynomialnd_permutations(d, n, coeffs, debugprint=debugprint, permutationmode=permutationmode, **kwargs)
    raise NoCriticalPointsError("Couldn't minimize polynomial under any permutation:\n{}".format(coeffs))

  permutation, permutationdict, result, figureofmerit = best

  #import pprint; pprint.pprint(signs)

  return result

def minimizepolynomialnd_permutationsasneeded(*args, **kwargs):
  return minimizepolynomialnd_permutations(*args, permutationmode="asneeded", **kwargs)

if __name__ == "__main__":
  coeffs = np.array([float(_) for _ in """
        5.49334216e-09 -9.84400122e-09  4.38160058e-07 -1.26382479e-07
       -9.12089215e-10  1.01516341e-06 -1.26332021e-07  1.09256020e-07
        1.63314502e-09  1.03372119e-05 -4.96514478e-06 -7.80328090e-08
        5.99469948e-07  2.18440354e-08  9.52325483e-09 -5.08274877e-07
        6.57523730e-06  6.04471018e-06 -1.30160136e-06 -6.39638559e-07
        4.03655366e-07  1.97068383e-08  1.61606392e-07 -1.96627744e-08
        0.00000000e+00  1.38450711e-05  9.09015592e-06 -1.71721919e-06
       -9.80367599e-06  8.96773536e-07  8.06458361e-07  1.43873072e-06
       -1.19835291e-07  0.00000000e+00  0.00000000e+00  4.06317330e-07
       -3.33848064e-07 -2.14655695e-07  9.67377758e-08  2.86534044e-06
        1.28511390e-06 -8.78287961e-07  5.24488740e-06 -1.27394601e-06
        1.19881998e-05 -5.41396188e-07 -4.49905521e-07  1.26380292e-07
        4.44793949e-07 -7.02872007e-08  4.20198939e-16  4.65950944e-08
       -8.35958714e-09  2.03796222e-07  0.00000000e+00  9.50188566e-07
        3.69815932e-06 -1.11180898e-06  1.87621678e-06 -9.39022434e-07
        1.35841041e-05 -3.41416561e-06  7.81379717e-07  3.12984343e-14
        0.00000000e+00  8.04957702e-07 -1.34019321e-07  1.47580088e-06
        0.00000000e+00  0.00000000e+00
  """.split()])

  import argparse
  p = argparse.ArgumentParser()
  p.add_argument("--verbose", action="store_true")
  args = p.parse_args()

  print minimizepolynomialnd_permutationsasneeded(4, 4, coeffs, verbose=True)

  #coeffs = coeffswithpermutedvariables(4, 4, coeffs, {"1": "z", "z": "1", "x": "x", "y": "y", "w": "w"})

  #print np.array(list(findcriticalpointspolynomialnd(4, 4, coeffs, **args.__dict__)))
