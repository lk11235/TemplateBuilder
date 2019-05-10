#!/usr/bin/env python

import abc, itertools, logging, sys

import numpy as np
import cvxpy as cp
from scipy import optimize

from polynomialalgebra import minimizepolynomialnd, minimizequadratic, minimizequartic

logger = logging.getLogger("cuttingplanemethod")

class CuttingPlaneMethodBase(object):
  __metaclass__ = abc.ABCMeta
  def __init__(self, x0, sigma, maxfractionaladjustment=0):
    if x0.shape != sigma.shape:
      raise ValueError("x0 and sigma have different shapes: {}, {}".format(x0.shape, sigma.shape))

    if len(x0.shape) == 1:
      x0 = np.array([[_] for _ in x0])
      sigma = np.array([[_] for _ in sigma])

    if len(x0) != self.xsize:
      raise ValueError("len(x0) should be {}, is actually {}".format(self.xsize, len(x0)))
    if len(sigma) != self.xsize:
      raise ValueError("len(sigma) should be {}, is actually {}".format(self.xsize, len(sigma)))

    self.__x0 = x0
    self.__sigma = sigma
    self.__constraints = []
    self.__results = None
    self.__maxfractionaladjustment = maxfractionaladjustment
    x = self.__x = cp.Variable(self.xsize)

    self.__loglikelihood = 0

    for x0column, sigmacolumn in itertools.izip(x0.T, sigma.T):
      shiftandscale = (self.__x - x0column) / sigmacolumn
      self.__loglikelihood += cp.quad_form(shiftandscale, np.diag([1]*self.xsize))

    self.__minimize = cp.Minimize(self.__loglikelihood)

  @abc.abstractproperty
  def xsize(self): "can just be a class member"

  @abc.abstractmethod
  def evalconstraint(self, potentialsolution):
    """
    Evaluates the potential solution to see if it satisfies the constraints.
    Should return two things:
     - the minimum value of the polynomial that has to be always positive
     - the values of the monomials at that minimum
       e.g. for a 4D quartic, (1, x1, x2, x3, x4, x1^2, x1x2, ..., x4^4)
    """

  def iterate(self):
    if self.__results is not None:
      raise RuntimeError("Can't iterate, already finished")

    if not self.__constraints:
      logger.info("x0:")
      logger.info(str(self.__x0))
      logger.info("sigma:")
      logger.info(str(self.__sigma))

    toprint = "starting iteration {}".format(len(self.__constraints)+1)
    logger.info("="*len(toprint))
    logger.info(toprint)
    logger.info("="*len(toprint))

    prob = cp.Problem(
      self.__minimize,
      self.__constraints,
    )

    solvekwargs = {
      "solver": cp.MOSEK,
    }
    try:
      prob.solve(**solvekwargs)
    except Exception as e:
      if "solve with verbose=True" in str(e):
        prob.solve(verbose=True, **solvekwargs)
      raise

    x = self.__x.value

    logger.info("found minimum {} at {}".format(prob.value, x))

    #does it satisfy the constraints?

    minimizepolynomial = self.evalconstraint(x)
    minvalue = minimizepolynomial.fun

    if minvalue >= 0:
      logger.info("Minimum of the constraint polynomial is %g --> finished successfully!", minvalue)
      self.__results = optimize.OptimizeResult(
        x=x,
        success=True,
        status=1,
        nit=len(self.__constraints)+1,
        maxcv=0,
        message="finished successfully",
        fun=prob.value
      )
    elif -minvalue < x[0] * self.__maxfractionaladjustment:
      logger.info("Minimum of the constraint polynomial is %g", minvalue)
      oldx0 = x[0]
      while minvalue < 0:
        print x[0], minvalue
        x[0] -= minvalue - np.finfo(float).eps
        minvalue = self.evalconstraint(x).fun
      logger.info("Multiply constant term by (1+%g) --> new minimum of the constraint polynomial is %g", x[0] / oldx0 - 1, minvalue)
      logger.info("Approximate minimum of the target function is {} at {}".format(self.__loglikelihood.value, x))
      self.__results = optimize.OptimizeResult(
        x=x,
        success=True,
        status=2,
        nit=len(self.__constraints)+1,
        maxcv=0,
        message="multiplied constant term by (1+{}) to get within constraint".format(x[0] / oldx0 - 1),
        fun=self.__loglikelihood.value
      )
    else:
      logger.info("Minimum of the constraint polynomial is %g --> adding a new constraint using this minimum", minvalue)
      self.__constraints.append(cp.matmul(minimizepolynomial.linearconstraint, self.__x) >= np.finfo(float).eps)

  def run(self, *args, **kwargs):
    while not self.__results: self.iterate(*args, **kwargs)
    return self.__results

class CuttingPlaneMethod1DQuadratic(CuttingPlaneMethodBase):
  xsize = 3
  evalconstraint = staticmethod(minimizequadratic)

class CuttingPlaneMethod1DQuartic(CuttingPlaneMethodBase):
  xsize = 5
  evalconstraint = staticmethod(minimizequartic)

class CuttingPlaneMethod4DQuadratic(CuttingPlaneMethodBase):
  xsize = 15
  def evalconstraint(self, coeffs):
    return minimizepolynomialnd(2, 4, coeffs)

class CuttingPlaneMethod4DQuartic(CuttingPlaneMethodBase):
  xsize = 70
  def evalconstraint(self, coeffs):
    return minimizepolynomialnd(4, 4, coeffs)

def cuttingplanemethod1dquadratic(*args, **kwargs):
  return CuttingPlaneMethod1DQuadratic(*args, **kwargs).run()
def cuttingplanemethod1dquartic(*args, **kwargs):
  return CuttingPlaneMethod1DQuartic(*args, **kwargs).run()
def cuttingplanemethod4dquadratic(*args, **kwargs):
  return CuttingPlaneMethod4DQuadratic(*args, **kwargs).run()
def cuttingplanemethod4dquartic(*args, **kwargs):
  return CuttingPlaneMethod4DQuartic(*args, **kwargs).run()

if __name__ == "__main__":
  logger.setLevel(logging.INFO)
  logger.addHandler(logging.StreamHandler(sys.stdout))
  a = np.array([1]*70)
  a[2] = -1
  print CuttingPlaneMethod4DQuartic(
    a,
    abs(a),
    maxfractionaladjustment=1e-6,
  ).run()
