import abc, array, collections, functools, itertools, re

import numpy as np
import uncertainties

import ROOT

TreeVariable = collections.namedtuple("TreeVariable", "formula nbins min max")

class TemplateComponentPieceBase(object):
  def __init__(self, name, printprefix, mirrortype, scaleby, xbins, ybins, zbins):
    self.__name = name
    self.__printprefix = printprefix

    self.__mirrortype = mirrortype
    assert uncertainties.std_dev(scaleby) == 0
    self.__scaleby = uncertainties.nominal_value(scaleby)

    self.__xbins = xbins
    self.__ybins = ybins
    self.__zbins = zbins

  @property
  def name(self): return self.__name
  @property
  def printprefix(self): return self.__printprefix
  @property
  def mirrortype(self): return self.__mirrortype
  @property
  def scaleby(self): return self.__scaleby
  @property
  def floor(self): return self.__floor

  @property
  def xbins(self): return self.__xbins
  @property
  def ybins(self): return self.__ybins
  @property
  def zbins(self): return self.__zbins
  @property
  def binsxyz(self):
    return itertools.product(xrange(1, self.xbins+1), xrange(1, self.ybins+1), xrange(1, self.zbins+1))

  @abc.abstractproperty
  def sumofallweights(self): pass
  @abc.abstractmethod
  def GetBinContentError(self, *args): pass
  @abc.abstractmethod
  def GetBinContentErrorAbs(self, *args): pass

class TemplateComponentPiece(TemplateComponentPieceBase):
  def __init__(
    self, name, printprefix,
    xtreeformula, xbins, xmin, xmax,
    ytreeformula, ybins, ymin, ymax,
    ztreeformula, zbins, zmin, zmax,
    cuttreeformula, weighttreeformula,
    mirrortype, scaleby,
  ):
    super(TemplateComponentPiece, self).__init__(name, printprefix, mirrortype, scaleby, xbins, ybins, zbins)

    self.__xtreeformula = xtreeformula
    self.__ytreeformula = ytreeformula
    self.__ztreeformula = ztreeformula
    self.__cuttreeformula = cuttreeformula
    self.__weighttreeformula = weighttreeformula

    self.__tdirectory = ROOT.gDirectory.GetDirectory(ROOT.gDirectory.GetPath())

    nameabs = name+"_absweights"
    nameallevents = name+"_allevents"

    hkey = self.__tdirectory.FindKey(name)
    habskey = self.__tdirectory.FindKey(nameabs)
    halleventskey = self.__tdirectory.FindKey(nameallevents)

    if hkey and habskey and halleventskey:
      self.__h = hkey.ReadObj()
      self.__habs = habskey.ReadObj()
      self.__hallevents = halleventskey.ReadObj()
      self.__h.SetDirectory(0)
      self.__habs.SetDirectory(0)
      self.__hallevents.SetDirectory(0)
      self.__locked = True
    else:
      self.__h = ROOT.TH3F(
        name, name,
        xbins, xmin, xmax,
        ybins, ymin, ymax,
        zbins, zmin, zmax,
      )

      self.__habs = ROOT.TH3F(
        nameabs, nameabs,
        xbins, xmin, xmax,
        ybins, ymin, ymax,
        zbins, zmin, zmax,
      )

      self.__hallevents = ROOT.TH1F(nameallevents, nameallevents, 1, 0, 1)

      self.__h.SetDirectory(0)
      self.__habs.SetDirectory(0)
      self.__hallevents.SetDirectory(0)

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

    if mirrortype is not None: assert ymin == -ymax and ybins%2 == 0

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

    weight = self.weight()
    if self.mirrortype is not None:
      weight /= 2
    weight *= self.scaleby

    if self.mirrortype is not None:
      sign = {"symmetric": 1, "antisymmetric": -1}[self.mirrortype]
      mirrorweight = sign*weight

    self.__hallevents.Fill(0, weight + (mirrorweight if self.mirrortype else 0))

    if self.passcut():
      binx = self.binx()
      biny = self.biny()
      binz = self.binz()

      if biny == 0: biny += 1e-10

      self.__h.Fill(binx, biny, binz, weight)
      self.__habs.Fill(binx, biny, binz, abs(weight))

      if self.mirrortype is not None:
        self.__h.Fill(binx, -biny, binz, mirrorweight)
        self.__habs.Fill(binx, -biny, binz, abs(mirrorweight))

  @property
  def integral(self):
    self.lock()
    error = array.array("d", [0])
    nominal = self.__h.IntegralAndError(1, self.__h.GetNbinsX(), 1, self.__h.GetNbinsY(), 1, self.__h.GetNbinsZ(), error)
    return uncertainties.ufloat(nominal, error[0])

  @property
  def sumofallweights(self):
    self.lock()
    return uncertainties.ufloat(self.__hallevents.GetBinContent(1), self.__hallevents.GetBinError(1))

  def GetBinContentError(self, *args):
    return uncertainties.ufloat(self.__h.GetBinContent(*args), self.__h.GetBinError(*args))
  def GetBinContentErrorAbs(self, *args):
    return uncertainties.ufloat(self.__habs.GetBinContent(*args), self.__habs.GetBinError(*args))

  def lock(self):
    if self.__locked: return

    if any(self.__habs.GetBinContent(x, y, z) for x, y, z in self.binsxyz):
      #floor the error
      maxerrorratio, errortoset = max(
        (self.__habs.GetBinError(x, y, z) / self.__habs.GetBinContent(x, y, z), self.__h.GetBinError(x, y, z))
          for x, y, z in self.binsxyz
        if self.__habs.GetBinError(x, y, z) != 0
      )
      #the reasoning being that if there's a bin with just one entry 2.3 +/- 2.3, then the zero bin could also have 2.3
      #but we can't draw that conclusion from a bin 1000 +/- 5.5

      for x, y, z in self.binsxyz:
        if self.__h.GetBinError(x, y, z) == 0:
          self.__h.SetBinError(x, y, z, errortoset)
          self.__habs.SetBinError(x, y, z, errortoset)

    self.__h.SetDirectory(self.__tdirectory)
    self.__habs.SetDirectory(self.__tdirectory)
    self.__hallevents.SetDirectory(self.__tdirectory)

    self.__locked = True

  @property
  def locked(self):
    return self.__locked

  @property
  def rootless(self):
    kwargs = {thing: getattr(self, thing) for thing in ("name", "printprefix", "mirrortype", "scaleby", "xbins", "ybins", "zbins")}
    kwargs["sumofallweights"] = self.sumofallweights
    kwargs["bincontents"] = {xyz: self.GetBinContentError(*xyz) for xyz in self.binsxyz}
    kwargs["bincontentsabs"] = {xyz: self.GetBinContentErrorAbs(*xyz) for xyz in self.binsxyz}
    return RootlessTemplateComponentPiece(**kwargs)

class RootlessTemplateComponentPiece(TemplateComponentPieceBase):
  def __init__(self, sumofallweights, bincontents, bincontentsabs, **kwargs):
    self.__sumofallweights = sumofallweights
    self.__bincontents = bincontents
    self.__bincontentsabs = bincontentsabs
    super(RootlessTemplateComponentPiece, self).__init__(**kwargs)
  @property
  def sumofallweights(self): return self.__sumofallweights
  def GetBinContentError(self, *args): return self.__bincontents[args]
  def GetBinContentErrorAbs(self, *args): return self.__bincontentsabs[args]

