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

  def makefinaltemplate(self, printbins):
    printbins = tuple(tuple(_) for _ in printbins)
    assert all(len(_) == 3 for _ in printbins)
    print
    print "Making the final template:"
    print "  {:40} {:45}".format(self.printprefix, self.name)
    print "from individual templates with integrals:"

    for component in self.__templatecomponents:
      component.lock()
      print "  {:45} {:10.3e}".format(component.name, component.integral)

    flooredbins = []
    outlierwarning = []
    printedbins = []

    for x, y, z in self.binsxyz:
      bincontent = {}
      for component in self.__templatecomponents:
        bincontent[component.name] = component.GetBinContentError(x, y, z)

      namestoremove = set()

      #remove outliers:
      #first try to use all the templatecomponents, then try one, then two, etc.
      for i in xrange(len(bincontent)-1):
        significances = {}
        #for each number: loop through the combinations of templatecomponents to possibly remove
        for namestomayberemove in itertools.combinations(bincontent, i):
          contentstomayberemove = tuple(bincontent[_] for _ in namestomayberemove)
          for name, content in bincontent.iteritems():
            #for each remaining templatecomponent, find the unbiased residual between its bin content
            #and the bin content predicted by the other remaining components
            if name in namestomayberemove: continue
            newunbiasedresidual = (
              content
              - weightedaverage(othercontent
                for othername, othercontent in bincontent.iteritems()
                if othername != name and othername not in namestomayberemove
              )
            )
            significance = abs(newunbiasedresidual.n) / newunbiasedresidual.s
            #if there's a 3sigma difference, then this combination of templatecomponents to remove is no good
            if significance > 3: break #out of the loop over remaining names
          else:
            #no remaining unbiased residuals are 3sigma
            #that means this combination of templatecomponents is a candidate to remove
            #if multiple combinations of the same number of templatecomponents fit this criterion,
            #then we pick the one that itself has the biggest normalized residual from the other templatecomponents
            #therefore we store it in significances
            if contentstomayberemove:
              unbiasedresidual = (
                weightedaverage(contentstomayberemove)
                - weightedaverage(othercontent for othername, othercontent in bincontent.iteritems() if othername not in namestomayberemove)
              )
              significances[namestomayberemove] = abs(unbiasedresidual.n) / unbiasedresidual.s
            else:
              significances[namestomayberemove] = float("inf")

        if significances:
          nameswithmaxsignificance, maxsignificance = max(significances.iteritems(), key=lambda x: x[1])
          namestoremove = nameswithmaxsignificance
          break

      if namestoremove:
        outlierwarning.append("  {:3d} {:3d} {:3d}: {}".format(x, y, z, ", ".join(sorted(namestoremove))))
      for name in namestoremove:
        del bincontent[name]

      if len(bincontent) < len(self.__templatecomponents) / 2.:
        raise RuntimeError("Removed more than half of the bincontents!  Please check.\n" + "\n".join("  {:45} {:10.3e}".format(component.name, component.GetBinContentError(x, y, z)) for component in self.__templatecomponents))

      if all(_.n == _.s == 0 for _ in bincontent.itervalues()):  #special case, empty histogram
        finalbincontent = bincontent.values()[0]
      else:                                                      #normal case
        finalbincontent = weightedaverage(bincontent.itervalues()) * self.__scaleby

      if (x, y, z) in printbins:
        thingtoprint = "  {:3d} {:3d} {:3d}:".format(x, y, z)
        fmt = "      {:<%d} {:10.3e}" % max(len(name) for name in bincontent)
        for name, content in bincontent.iteritems():
          thingtoprint += "\n"+fmt.format(name, content)
        thingtoprint += "\n"+fmt.format("final", finalbincontent)
        printedbins.append(thingtoprint)

      self.__h.SetBinContent(x, y, z, finalbincontent.nominal_value)
      self.__h.SetBinError(x, y, z, finalbincontent.std_dev)

    if outlierwarning:
      print
      print "Warning: there are outliers in some bins:"
      for _ in outlierwarning: print _

    if printedbins:
      print
      print "Bins you requested to print:"
      for _ in printedbins: print _

    if self.__mirrortype is not None: self.__domirror()
    if self.__floor is not None: self.__dofloor()

    print
    print "final integral = {:10.3e}".format(self.integral)
    print
