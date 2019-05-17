#!/usr/bin/env python

import abc, collections, cStringIO, itertools, logging, sys

import numpy as np
import cvxpy as cp
from scipy import optimize

from polynomialalgebra import getpolynomialndmonomials, minimizepolynomialnd, minimizequadratic, minimizequartic

logger = logging.getLogger("cuttingplanemethod")

class CuttingPlaneMethodBase(object):
  __metaclass__ = abc.ABCMeta
  def __init__(self, x0, sigma, maxfractionaladjustment=0, reportdeltafun=True, printlogaterror=True, moreargsforevalconstraint=()):
    x0 = np.array(x0)
    sigma = np.array(sigma)

    if x0.shape != sigma.shape:
      raise ValueError("x0 and sigma have different shapes: {}, {}".format(x0.shape, sigma.shape))

    if len(x0.shape) == 1:
      x0 = np.array([[_] for _ in x0])
      sigma = np.array([[_] for _ in sigma])

    if len(x0) != self.xsize:
      raise ValueError("len(x0) should be {}, is actually {}".format(self.xsize, len(x0)))
    if len(sigma) != self.xsize:
      raise ValueError("len(sigma) should be {}, is actually {}".format(self.xsize, len(sigma)))

    self.__cuttingplanes = []
    self.__otherconstraints = []
    self.__moreargsforevalconstraint = moreargsforevalconstraint
    self.__results = None
    self.__maxfractionaladjustment = maxfractionaladjustment

    self.__reportdeltafun = reportdeltafun
    self.__funatminimum = 0

    self.__printlogaterror = printlogaterror
    if printlogaterror:
      self.__logstream = cStringIO.StringIO()
      self.__logstreamhandler = logging.StreamHandler(self.__logstream)
      logger.addHandler(self.__logstreamhandler)
      logger.setLevel(logging.INFO)

    self.__initminimization(x0, sigma)

  def __initminimization(self, x0, sigma):
    x = self.__x = cp.Variable(self.xsize)
    self.__t = None

    #===================================================================
    #simplest approach:
    #for x0column, sigmacolumn in itertools.izip(x0.T, sigma.T):
    #  shiftandscale = (self.__x - x0column) / sigmacolumn
    #  self.__tominimize += cp.quad_form(shiftandscale, np.diag([1]*self.xsize))
    #this is inefficient because the code doesn't treat a sum of quad_forms nicely
    #===================================================================

    #===================================================================
    #second approach:
    #shiftandscale_quadraticterm = shiftandscale_linearterm = shiftandscale_constantterm = 0
    #for x0column, sigmacolumn in itertools.izip(x0.T, sigma.T):
    #  shiftandscale_quadraticterm += np.diag(1 / sigmacolumn**2)
    #  shiftandscale_linearterm += -2 * x0column / sigmacolumn**2
    #  shiftandscale_constantterm += sum(x0column**2 / sigmacolumn**2)

    #quadraticterm = cp.quad_form(x, shiftandscale_quadraticterm)
    #linearterm = cp.matmul(shiftandscale_linearterm, x)
    #constantterm = shiftandscale_constantterm
    #self.__tominimize = quadraticterm + linearterm + constantterm
    #===================================================================

    #===================================================================
    #third approach, starts out the same as the second with a change in notation
    Q = c = r = 0
    for x0column, sigmacolumn in itertools.izip(x0.T, sigma.T):
      #note the 2 here! it's to get the 1/2 in 1/2 x^T Q x + c^T x + r
      Q += 2 * np.diag(1 / sigmacolumn**2)
      c += -2 * x0column / sigmacolumn**2
      r += sum(x0column**2 / sigmacolumn**2)
    #we are minimizing 1/2 x^T Q x + c^T x + r, as in https://docs.mosek.com/modeling-cookbook/cqo.html#equation-eq-cqo-qcqo

    #now we want to scale the coefficients to get numbers even closer to 1.
    #We could scale all the coefficients to have sigma=1.  This wouldn't help however,
    #because we would have to scale them back in order to evaluate the nd polynomial
    #and we'd get a linear constraint with big numbers, same as before.
    #Instead, we want to scale the _variables_ of the nd polynomial, or in other words
    #scale the coefficients according to how they multiply those variables.
    #This doesn't affect whether the polynomial ever goes negative.

    logger.info("x0:")
    logger.info(str(x0))
    logger.info("sigma:")
    logger.info(str(sigma))
    logger.info("Q matrix:")
    logger.info(str(np.diag(Q)))
    logger.info("linear coefficients:")
    logger.info(str(c))

    multiplyvariables, multiplycoeffs = self.__findmultiplycoeffs(np.diag(Q) ** .5)
    self.__multiplycoeffs = multiplycoeffs
    Q = np.diag(np.diag(Q) / multiplycoeffs**2)
    c /= multiplycoeffs

    self.__multipliedargsforevalconstraint = tuple(
      f(multiplyvariables, arg) for f, arg in itertools.izip(self.modifymoreargs(), self.__moreargsforevalconstraint)
    )

    logger.info("Multiplied variables to get coefficients closer to 1 for minimization.")
    logger.info("Q matrix:")
    logger.info(str(np.diag(Q)))
    logger.info("linear coefficients:")
    logger.info(str(c))

    t = self.__t = cp.Variable()
    self.__tominimize = 0.5 * cp.quad_form(x, Q) + cp.matmul(c, x)
    #===================================================================

    self.__minimize = cp.Minimize(self.__tominimize)

    self.__otherconstraints += [
      self.__x[i]>=0 for i in self.maxpowerindices
    ]

  @property
  def polynomialvariables(self):
    return sorted(sum(self.monomials, collections.Counter()).keys()) #including '1'

  @staticmethod
  def modifymoreargs():
    return ()

  @property
  def maxpowerindices(self):
    maxpowers = {
      varname: max(monomial[varname] for monomial in self.monomials)
        for varname in self.polynomialvariables
    }
    for v, p in maxpowers.iteritems():
      if p%2: raise ValueError("max power of {} is odd: {}".format(v, p))
    result = []
    for i, monomial in enumerate(self.monomials):
      if any(monomial[v] == p for v, p in maxpowers.iteritems()):
        result.append(i)
    return result

  def __findmultiplycoeffs(self, diagF, verbose=False):
    monomials = self.monomials

    logdiagF = np.log(diagF)
    logmultiplyvariables = collections.defaultdict(cp.Variable)

    tominimize = 0
    for monomial, logoneovercoeff in itertools.izip_longest(monomials, logdiagF):
      tominimize += (logoneovercoeff + sum(logmultiplyvariables[variable] for variable in monomial.elements()))**2
    minimize = cp.Minimize(tominimize)
    prob = cp.Problem(minimize)
    prob.solve(verbose=verbose)

    logmultiplycoeffs = -np.array([sum(logmultiplyvariables[variable].value for variable in monomial.elements()) for monomial in monomials])
    multiplyvariables = {k: np.exp(v.value) for k, v in logmultiplyvariables.iteritems()}
    multiplycoeffs = np.exp(logmultiplycoeffs)

    if verbose:
      print {k: v.value for k, v in logmultiplyvariables.iteritems()}
      print logdiagF
      for i, logcoeff in enumerate(logmultiplycoeffs):
        #subtract because diagF is proportional to 1/coeff
        logdiagF[i] -= sum(logmultiplyvariables[variable].value for variable in monomial.elements())
      print logdiagF
      print diagF
      print np.exp(logdiagF)

    return multiplyvariables, multiplycoeffs

  def __del__(self):
    if self.__printlogaterror:
      logger.handlers.remove(self.__logstreamhandler)

  @abc.abstractproperty
  def xsize(self): "can just be a class member"

  @abc.abstractmethod
  def evalconstraint(self, potentialsolution):
    """
    Evaluates the potential solution to see if it satisfies the constraints.
    Should return an OptimizeResult that includes:
     - the minimum value of the polynomial that has to be always positive
     - the values of the monomials at that minimum
       e.g. for a 4D quartic, (1, x1, x2, x3, x4, x1^2, x1x2, ..., x4^4)
    """
  @abc.abstractproperty
  def monomials(self):
    """
    Order of monomials in the polynomial, corresponding to the expected order of coefficients
    """

  def iterate(self):
    if self.__results is not None:
      raise RuntimeError("Can't iterate, already finished")

    toprint = "starting iteration {}".format(len(self.__cuttingplanes)+1)
    logger.info("="*len(toprint))
    logger.info(toprint)
    logger.info("="*len(toprint))

    prob = cp.Problem(
      self.__minimize,
      self.__otherconstraints + self.__cuttingplanes,
    )

    solvekwargs = {
      "solver": cp.MOSEK,
    }
    try:
      prob.solve(**solvekwargs)
      x = self.__x.value

      if self.__reportdeltafun and not self.__cuttingplanes:
        self.__funatminimum = prob.value

      logger.info("found minimum {} at:\n{}".format(prob.value - self.__funatminimum, x))

      #does it satisfy the constraints?

      minimizepolynomial = self.evalconstraint(x, *self.__moreargsforevalconstraint)
      minvalue = minimizepolynomial.fun
    except BaseException as e:
      if self.__printlogaterror:
        print self.__logstream.getvalue()
      prob.solve(verbose=True, **solvekwargs)
      raise

    if minvalue >= 0:
      logger.info("Minimum of the constraint polynomial is %g --> finished successfully!", minvalue)
      self.__results = optimize.OptimizeResult(
        x=x / self.__multiplycoeffs,
        success=True,
        status=1,
        nit=len(self.__cuttingplanes)+1,
        maxcv=0,
        message="finished successfully",
        fun=prob.value - self.__funatminimum
      )
      return

    if -minvalue < x[0] * self.__maxfractionaladjustment:
      logger.info("Minimum of the constraint polynomial is %g", minvalue)

      oldx0 = x[0]
      multiplier = 1
      while minvalue < 0:
        lastx0 = x[0]
        print x[0], minvalue
        x[0] -= minvalue - multiplier*np.finfo(float).eps
        if x[0] == lastx0: multiplier += 1
        minvalue = self.evalconstraint(x).fun

      if x[0] / oldx0 - 1 < self.__maxfractionaladjustment:
        logger.info("Multiply constant term by (1+%g) --> new minimum of the constraint polynomial is %g", x[0] / oldx0 - 1, minvalue)
        logger.info("Approximate minimum of the target function is {} at {}".format(self.__tominimize.value - self.__funatminimum, x))
        self.__results = optimize.OptimizeResult(
          x=x / self.__multiplycoeffs,
          success=True,
          status=2,
          nit=len(self.__cuttingplanes)+1,
          maxcv=0,
          message="multiplied constant term by (1+{}) to get within constraint".format(x[0] / oldx0 - 1),
          fun=self.__tominimize.value - self.__funatminimum
        )
        return

    logger.info("Minimum of the constraint polynomial is {} at {} --> adding a new constraint using this minimum:\n{}".format(minvalue, minimizepolynomial.x, minimizepolynomial.linearconstraint))
    self.__cuttingplanes.append(
      cp.matmul(
        minimizepolynomial.linearconstraint[self.useconstraintindices,],
        self.__x
      ) >= np.finfo(float).eps
    )

  useconstraintindices = slice(None, None, None)

  def run(self, *args, **kwargs):
    while not self.__results: self.iterate(*args, **kwargs)
    return self.__results

class CuttingPlaneMethod1DQuadratic(CuttingPlaneMethodBase):
  xsize = 3
  monomials = list(getpolynomialndmonomials(2, 1))
  evalconstraint = staticmethod(minimizequadratic)

class CuttingPlaneMethod1DQuartic(CuttingPlaneMethodBase):
  xsize = 5
  monomials = list(getpolynomialndmonomials(4, 1))
  evalconstraint = staticmethod(minimizequartic)

class CuttingPlaneMethod3DQuadratic(CuttingPlaneMethodBase):
  xsize = 10
  monomials = list(getpolynomialndmonomials(2, 3))
  def evalconstraint(self, coeffs):
    return minimizepolynomialnd(2, 3, coeffs)

class CuttingPlaneMethod4DQuadratic(CuttingPlaneMethodBase):
  xsize = 15
  monomials = list(getpolynomialndmonomials(2, 4))
  def evalconstraint(self, coeffs):
    return minimizepolynomialnd(2, 4, coeffs)

class CuttingPlaneMethod4DQuartic(CuttingPlaneMethodBase):
  xsize = 70
  monomials = list(getpolynomialndmonomials(4, 4))
  def evalconstraint(self, coeffs):
    return minimizepolynomialnd(4, 4, coeffs)

class CuttingPlaneMethod4DQuartic_4thVariableQuadratic(CuttingPlaneMethodBase):
  xsize = 65
  def insertzeroatindices():
    for idx, variables in enumerate(getpolynomialndmonomials(4, 4)):
      if variables["z"] >= 3:
        yield idx
  insertzeroatindices = list(insertzeroatindices())

  useconstraintindices = range(70)
  monomials = list(getpolynomialndmonomials(4, 4))
  for _ in sorted(insertzeroatindices, reverse=True):
    assert useconstraintindices[_] == _
    del useconstraintindices[_]
    del monomials[_]
  del _

  def evalconstraint(self, coeffs):
    coeffs = iter(coeffs)
    newcoeffs = np.array([0 if i in self.insertzeroatindices else next(coeffs) for i in xrange(70)])
    for remaining in coeffs: assert False
    return minimizepolynomialnd(4, 4, newcoeffs)

def cuttingplanemethod1dquadratic(*args, **kwargs):
  return CuttingPlaneMethod1DQuadratic(*args, **kwargs).run()
def cuttingplanemethod1dquartic(*args, **kwargs):
  return CuttingPlaneMethod1DQuartic(*args, **kwargs).run()
def cuttingplanemethod3dquadratic(*args, **kwargs):
  return CuttingPlaneMethod3DQuadratic(*args, **kwargs).run()
def cuttingplanemethod4dquadratic(*args, **kwargs):
  return CuttingPlaneMethod4DQuadratic(*args, **kwargs).run()
def cuttingplanemethod4dquartic(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic(*args, **kwargs).run()
def cuttingplanemethod4dquartic_4thvariablequadratic(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic_4thVariableQuadratic(*args, **kwargs).run()

class CuttingPlaneMethod4DQuartic_4thVariableSmallBeyondQuadratic_Step2(CuttingPlaneMethodBase):
  z34indices = [i for i, monomial in enumerate(getpolynomialndmonomials(4, 4)) if monomial["z"] >= 3]
  xsize = 5
  assert len(z34indices) == xsize

  useconstraintindices = range(70)
  monomials = list(getpolynomialndmonomials(4, 4))
  unusedmonomials = []
  for _ in sorted(range(70), reverse=True):
    if _ in z34indices: continue
    assert useconstraintindices[_] == _
    del useconstraintindices[_]
    unusedmonomials.insert(0, monomials.pop(_))
  del _

  def evalconstraint(self, coeffs, othercoeffs):
    coeffs = iter(coeffs)
    othercoeffs = iter(othercoeffs)
    newcoeffs = np.array([next(coeffs) if i in self.z34indices else next(othercoeffs) for i in xrange(70)])
    for remaining in coeffs: assert False
    for remaining in othercoeffs: assert False
    return minimizepolynomialnd(4, 4, newcoeffs)

  @property
  def maxpowerindices(self):
    maxpowers = {
      varname: max(monomial[varname] for monomial in self.monomials)
        for varname in self.polynomialvariables
        if varname == "z"
    }
    for v, p in maxpowers.iteritems():
      if p%2: raise ValueError("max power of {} is odd: {}".format(v, p))
    result = []
    for i, monomial in enumerate(self.monomials):
      if any(monomial[v] == p for v, p in maxpowers.iteritems()):
        result.append(i)
    return result

  @classmethod
  def modifymoreargs(cls):
    def modifyothercoeffs(multiplyvariables, othercoeffs):
      multiplyothercoeffs = 1/np.array([np.prod([multiplyvariables[variable] for variable in monomial.elements()]) for monomial in cls.unusedmonomials])
      return multiplyothercoeffs * othercoeffs
    return modifyothercoeffs,

def cuttingplanemethod4dquartic_4thvariablezerobeyondquadratic(x0, sigma, *args, **kwargs):
  z34indices = [i for i, monomial in enumerate(getpolynomialndmonomials(4, 4)) if monomial["z"] >= 3]

  assert np.all(x0[z34indices] == 0)

  x0withoutz34 = np.array([_ for i, _ in enumerate(x0) if i not in z34indices])
  sigmawithoutz34 = np.array([_ for i, _ in enumerate(sigma) if i not in z34indices])

  result = cuttingplanemethod4dquartic_4thvariablequadratic(x0withoutz34, sigmawithoutz34, *args, **kwargs)

  x = iter(result.x)
  result.x = np.array([0 if i in z34indices else next(x) for i in xrange(len(x0))])
  for remaining in x: assert False

  result.message += " (4th variable is only quadratic)"

  return result

def cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic(x0, sigma, *args, **kwargs):
  z34indices = [i for i, monomial in enumerate(getpolynomialndmonomials(4, 4)) if monomial["z"] >= 3]

  x0withoutz34 = np.array([_ for i, _ in enumerate(x0) if i not in z34indices])
  sigmawithoutz34 = np.array([_ for i, _ in enumerate(sigma) if i not in z34indices])
  x0z34 = np.array([_ for i, _ in enumerate(x0) if i in z34indices])
  sigmaz34 = np.array([_ for i, _ in enumerate(sigma) if i in z34indices])

  result1 = cuttingplanemethod4dquartic_4thvariablequadratic(x0withoutz34, sigmawithoutz34, *args, **kwargs)
  result2 = cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic_step2(x0z34, sigmaz34, *args, moreargsforevalconstraint=(result1.x,))

  result = optimize.OptimizeResult({
    k+"_step1": v for k, v in result1.iteritems()
  })
  result.update({
    k+"_step2": v for k, v in result2.iteritems()
  })
  x1 = iter(result1.x)
  x2 = iter(result2.x)
  result.x = np.array([next(x2) if i in z34indices else next(x1) for i in xrange(len(x0))])
  for remaining in x1: assert False
  for remaining in x2: assert False

  result.message = "4th variable is small beyond quadratic, 2 steps.\nFirst: {}\nSecond: {}".format(result1.message, result2.message)
  result.nit = result1.nit + result2.nit
  result.maxcv = result1.maxcv + result2.maxcv
  result.fun = result1.fun + result2.fun
  result.status = max(result1.status, result2.status)

  return result

def cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic_step2(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic_4thVariableSmallBeyondQuadratic_Step2(*args, **kwargs).run()

if __name__ == "__main__":
  logger.setLevel(logging.INFO)
  logger.addHandler(logging.StreamHandler(sys.stdout))
  a = np.array([[1, 2.]]*70)
  a[2,:] *= -1
  print CuttingPlaneMethod4DQuartic(
    a,
    abs(a),
    maxfractionaladjustment=1e-6,
  ).run()
