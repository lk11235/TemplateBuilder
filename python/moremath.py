#!/usr/bin/env python

import functools
import itertools

import autograd
import autograd.numpy as np
import scipy.optimize as optimize
import uncertainties

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
      coeff * np.prod((coeff,) + xs)
      for coeff, xs in itertools.izip_longest(
        coeffs,
        itertools.combinations_with_replacement(xand1, 4),
      )
    )
  return quartic4d

def minimizequartic4d(coeffs):
  quartic = getquartic4d(coeffs)
  quarticjacobian = autograd.jacobian(quartic)
  quartichessian = autograd.hessian_vector_product(quartic)
  result = optimize.basinhopping(
    func=quartic,
    x0=np.array([1., 1., -5., 3.]),
    minimizer_kwargs=dict(
      jac=quarticjacobian,
      hess=quartichessian,
    ),
  )
  print
  return result

if __name__ == "__main__":
  a = np.array([1 if i in (0, 69, 35, 55, 65) else -1 if i in (5,) else 0 for i in xrange(70)])
  print minimizequartic4d(a)
