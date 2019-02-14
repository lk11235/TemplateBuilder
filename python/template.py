import array
import itertools

from collections import Counter

import uncertainties

from moreuncertainties import weightedaverage

class Template(object):
  def __init__(
    self, name, trees,
    xformula, xbins, xmin, xmax,
    yformula, ybins, ymin, ymax,
    zformula, zbins, zmin, zmax,
    cutformula, weightformula,
    mirrortype, scaleby, floor,
  ):
    self.__name = name
    self.__templatecomponenthandles = [
      tree.registertemplatecomponent(
        name+"_"+str(i),
        xformula, xbins, xmin, xmax,
        yformula, ybins, ymin, ymax,
        zformula, zbins, zmin, zmax,
        cutformula, weightformula,
      )
      for i, tree in enumerate(trees)
    ]

    import ROOT

    self.__h = ROOT.TH3F(
      name, name,
      xbins, xmin, xmax,
      ybins, ymin, ymax,
      zbins, zmin, zmax,
    )

    self.__xbins = xbins
    self.__ybins = ybins
    self.__zbins = zbins

    self.__mirrortype = mirrortype
    if mirrortype not in (None, "symmetric", "antisymmetric"):
      raise ValueError("invalid mirrortype {}: has to be None, symmetric, or antisymmetric".format(mirrortype))

    if scaleby is None: scaleby = 1
    self.__scaleby = scaleby

    self.__floor = floor

  @property
  def name(self): return self.__name

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

  def __domirror(self):
    for x, y, z in self.binsxyz:
      if y > self.__ybins / 2: continue
      sign = {"symmetric": 1, "antisymmetric": -1}[self.__mirrortype]
      newbincontent = weightedaverage((
        self.GetBinContentError(x, y, z),
        sign * self.GetBinContentError(x, self.__ybins+1-y, z),
      ))

      self.__h.SetBinContent(x, y, z, newbincontent.nominal_value)
      self.__h.SetBinError(x, y, z, newbincontent.std_dev)

      self.__h.SetBinContent(x, self.__ybins+1-y, z, sign*newbincontent.nominal_value)
      self.__h.SetBinError(x, self.__ybins+1-y, z, newbincontent.std_dev)

  def __dofloor(self):
    floor = self.__floor
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

  @property
  def __templatecomponents(self):
    return [handle() for handle in self.__templatecomponenthandles]

  def makefinaltemplate(self):
    print
    print "Making the final template:"
    print "  "+self.name
    print "from individual templates with integrals:"

    for component in self.__templatecomponents:
      component.lock()
      print "  {:20} {:8.3e}".format(component.name, component.integral)

    flooredbins = []

    outliers = Counter()

    for x, y, z in self.binsxyz:
      bincontent = {}
      for component in self.__templatecomponents:
        bincontent[component.name] = component.GetBinContentError(x, y, z)

      while True:
        averagebincontent = weightedaverage(bincontent.itervalues())
        maxdifferencesignificance, namewithmaxdifference = max(
          (abs((content - averagebincontent).n / (content - averagebincontent).s), name)
          for name, content in bincontent.items()
        )
        if maxdifferencesignificance > 3:
          outliers[namewithmaxdifference] += 1
          del bincontent[namewithmaxdifference]
        else:
          break

      if len(bincontent) < len(self.__templatecomponents) / 2.:
        raise RuntimeError("Removed more than half of the bincontents!  Please check.\n" + "\n".join("  {:20} {:8.3e}".format(component.name, component.GetBinContentError(x, y, z)) for component in self.__templatecomponents))

      finalbincontent = weightedaverage(bincontent.itervalues()) * self.__scaleby
      self.__h.SetBinContent(x, y, z, finalbincontent.nominal_value)
      self.__h.SetBinError(x, y, z, finalbincontent.std_dev)

    if outliers: print self.name + ": the following samples had outliers in some bins: " + ", ".join("{} ({})".format(k, v) for k, v in outliers.iteritems())

    if self.__mirrortype is not None: self.__domirror()
    if self.__floor is not None: self.__dofloor()

    print "final integral = {:8.3e}".format(self.integral)

    print
