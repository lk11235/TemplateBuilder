#!/usr/bin/env python

import abc

import numpy as np
import cvxpy as cp
from scipy import optimize

from polynomialalgebra import minimizepolynomialnd, minimizequadratic, minimizequartic

class CuttingPlaneMethodBase(object):
  __metaclass__ = abc.ABCMeta
  def __init__(self, x0, sigma, verbose=False):
    if len(x0) != self.xsize:
      raise ValueError("len(x0) should be {}, is actually {}".format(self.xsize, len(x0)))
    if len(sigma) != self.xsize:
      raise ValueError("len(sigma) should be {}, is actually {}".format(self.xsize, len(sigma)))

    self.__x0 = x0
    self.__sigma = sigma
    self.__verbose = verbose
    self.__constraints = []
    self.__results = None
    x = self.__x = cp.Variable(self.xsize)

    shiftandscale = (self.__x - self.__x0) / self.__sigma
    print shiftandscale.curvature
#    loglikelihood = cp.matmul(shiftandscale, shiftandscale)
    loglikelihood = cp.quad_form(shiftandscale, np.diag([1]*self.xsize))
    print loglikelihood.curvature
    self.__minimize = cp.Minimize(loglikelihood)

  @property
  def verbose(self): return self.__verbose

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

    if self.verbose:
      toprint = "starting iteration {}".format(len(self.__constraints)+1)
      print "="*len(toprint)
      print toprint
      print "="*len(toprint)

    prob = cp.Problem(
      self.__minimize,
      self.__constraints,
    )
    prob.solve()

    if self.verbose:
      print "found minimum", prob.value, "at", self.__x.value

    #does it satisfy the constraints?

    minimizepolynomial = self.evalconstraint(self.__x.value)
    minvalue = minimizepolynomial.fun

    if minvalue >= 0:
      if self.verbose:
        print "Minimum of the constraint polynomial is", minvalue, " --> finished successfully!"
      self.__results = optimize.OptimizeResult(
        x=self.__x.value,
        success=True,
        status=1,
        nit=len(self.__constraints)+1,
        maxcv=0,
        message="finished successfully",
        fun=prob.value
      )
    else:
      if self.verbose:
        print "Minimum of the constraint polynomial is", minvalue, " --> adding a new constraint using this minimum"
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

if __name__ == "__main__":
  a = np.array([1]*70)
  a[2] = -1
  print CuttingPlaneMethod4DQuartic(
    a,
    a,
    verbose=True
  ).run()
