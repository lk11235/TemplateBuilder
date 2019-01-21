import collections
import functools
import re

import ROOT

TreeVariable = collections.namedtuple("TreeVariable", "formula nbins min max")

class TemplateComponent(object):
  def __init__(
    self, name,
    xtreeformula, xbins, xmin, xmax,
    ytreeformula, ybins, ymin, ymax,
    ztreeformula, zbins, zmin, zmax,
    cuttreeformula, weighttreeformula,
  ):
    self.__name = name

    self.__xtreeformula = xtreeformula
    self.__ytreeformula = ytreeformula
    self.__ztreeformula = ztreeformula
    self.__cuttreeformula = cuttreeformula
    self.__weighttreeformula = weighttreeformula

    self.__h = ROOT.TH3F(
      name, name,
      xbins, xmin, xmax,
      ybins, ymin, ymax,
      zbins, zmin, zmax,
    )

    self.__forcewithinlimitsx = functools.partial(
      self.forcewithinlimits,
      xmin + (xmax - xmin) / xbins / 10, 
      xmax - (xmax - xmin) / xbins / 10, 
    )
    self.__forcewithinlimitsy = functools.partial(
      self.forcewithinlimits,
      ymin + (ymax - ymin) / ybins / 10, 
      ymax - (ymax - ymin) / ybins / 10, 
    )
    self.__forcewithinlimitsz = functools.partial(
      self.forcewithinlimits,
      zmin + (zmax - zmin) / zbins / 10, 
      zmax - (zmax - zmin) / zbins / 10, 
    )

    self.__locked = False

  @staticmethod
  def forcewithinlimits(lower, upper, value):
    return min(max(value, lower), upper)

  def binx(self):
    return self.__forcewithinlimitsx(self.__xtreeformula.EvalInstance())
  def biny(self):
    return self.__forcewithinlimitsy(self.__ytreeformula.EvalInstance())
  def binz(self):
    return self.__forcewithinlimitsz(self.__ztreeformula.EvalInstance())
  def weight(self):
    return self.__weighttreeformula.EvalInstance()
  def passcut(self):
    return self.__cuttreeformula.EvalInstance()

  def fill(self):
    if self.__locked:
      raise ValueError("Can't fill {} after it's locked".format(self))
    if self.passcut():
      self.__h.Fill(self.binx(), self.biny(), self.binz(), self.weight())

  def GetBinContentError(self, *args):
    return uncertainties.ufloat(self.__h.GetBinContent(*args), self.__h.GetBinError(*args))

  def lock(self):
    self.__locked = True
