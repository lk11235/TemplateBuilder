import array
import itertools
import os

from collections import Counter

import uncertainties

from moreuncertainties import weightedaverage

class Template(object):
  def __init__(
    self, name, printprefix, trees,
    xformula, xbins, xmin, xmax,
    yformula, ybins, ymin, ymax,
    zformula, zbins, zmin, zmax,
    cutformula, weightformula,
    mirrortype, scaleby, floor,
  ):
    filenames = [tree.filename for tree in trees]
    commonprefix = os.path.commonprefix(filenames)
    commonsuffix = os.path.commonprefix(list(_[::-1] for _ in filenames))[::-1]

    self.__name = name
    self.__printprefix = printprefix
    self.__templatecomponenthandles = [
      tree.registertemplatecomponent(
        name+"_"+tree.filename.replace(commonprefix, "", 1)[::-1].replace(commonsuffix[::-1], "", 1)[::-1], printprefix,
        xformula, xbins, xmin, xmax,
        yformula, ybins, ymin, ymax,
        zformula, zbins, zmin, zmax,
        cutformula, weightformula,
      )
      for i, tree in enumerate(trees)
    ]

    import ROOT

    self.__tdirectory = ROOT.gDirectory.GetDirectory(ROOT.gDirectory.GetPath())

    self.__h = ROOT.TH3F(
      name, name,
      xbins, xmin, xmax,
      ybins, ymin, ymax,
      zbins, zmin, zmax,
    )

    self.__h.SetDirectory(0)

    self.__xbins = xbins
    self.__ybins = ybins
    self.__zbins = zbins

    self.__mirrortype = mirrortype
    if mirrortype not in (None, "symmetric", "antisymmetric"):
      raise ValueError("invalid mirrortype {}: has to be None, symmetric, or antisymmetric".format(mirrortype))

    if scaleby is None: scaleby = 1
    self.__scaleby = scaleby

    self.__floor = floor

    self.__finalized = self.__didscale = self.__didmirror = self.__didfloor = False

  @property
  def name(self): return self.__name
  @property
  def printprefix(self): return self.__printprefix

  @property
  def binsxyz(self):
    return itertools.product(xrange(1, self.__xbins+1), xrange(1, self.__ybins+1), xrange(1, self.__zbins+1))

  @property
  def integral(self):
    error = array.array("d", [0])
    nominal = self.__h.IntegralAndError(1, self.__h.GetNbinsX(), 1, self.__h.GetNbinsY(), 1, self.__h.GetNbinsZ(), error)
    return uncertainties.ufloat(nominal, error[0])

  def GetBinContentError(self, *args):
    return uncertainties.ufloat(self.__h.GetBinContent(*args), self.__h.GetBinError(*args))

  def SetBinContentError(self, *args):
    if self.finalized:
      raise RuntimeError("Can't set bin content after it's finalized")
    self.__h.SetBinContent(*args[:-1]+(uncertainties.nominal_value(args[-1]),))
    self.__h.SetBinError(*args[:-1]+(uncertainties.std_dev(args[-1].s),))

  def doscale(self):
    if self.__didscale: raise RuntimeError("Trying to scale twice!")
    self.__didscale = True
    self.__h.Scale(self.__scaleby)

  def domirror(self):
    if self.__didmirror: raise RuntimeError("Trying to mirror twice!")
    self.__didmirror = True
    if self.__mirrortype is None: return
    for x, y, z in self.binsxyz:
      if y > self.__ybins / 2: continue
      sign = {"symmetric": 1, "antisymmetric": -1}[self.__mirrortype]
      newbincontent = weightedaverage((
        self.GetBinContentError(x, y, z),
        sign * self.GetBinContentError(x, self.__ybins+1-y, z),
      ))

      self.SetBinContentError(x, y, z, newbincontent)
      self.SetBinContentError(x, self.__ybins+1-y, z, sign*newbincontent)

  def dofloor(self):
    if self.__didfloor: raise RuntimeError("Trying to floor twice!")
    self.__didfloor = True
    floor = self.__floor
    if floor is None: return
    if floor.nominal_value <= 0:
      raise ValueError("Invalid floor {}: has to be positive.".format(floor.nominal_value))

    if floor.std_dev == 0:
      #use this procedure to estimate the error for floored bins
      maxerrorratio, errortoset = max(
        (self.__h.GetBinError(x, y, z) / self.__h.GetBinContent(x, y, z), self.__h.GetBinError(x, y, z))
          for x, y, z in self.binsxyz
        if self.__h.GetBinContent(x, y, z) != 0
      )
      #the reasoning being that if there's a bin with just one entry 2.3 +/- 2.3, then the zero bin could also have 2.3
      #but we can't draw that conclusion from a bin 1000 +/- 5.5

      floor = uncertainties.ufloat(floor.nominal_value, errortoset)

    for x, y, z in self.binsxyz:
      if self.__h.GetBinContent(x, y, z) <= floor.nominal_value:
        self.__h.SetBinContent(x, y, z, floor.nominal_value)
        self.__h.SetBinError(x, y, z, floor.std_dev)

  def finalize(self):
    self.doscale()
    self.domirror()
    self.dofloor()
    self.__finalized = True
    self.__h.SetDirectory(self.__tdirectory)

  @property
  def finalized(self):
    return self.__finalized

  @property
  def templatecomponents(self):
    return [handle() for handle in self.__templatecomponenthandles]
