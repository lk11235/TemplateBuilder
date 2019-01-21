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
    self.__templatecomponents = [
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
    self.__scaleby = scaleby

    self.__floor = floor

  @property
  def binsxyz(self):
    return itertools.product(xrange(1, self.__xbins+1), xrange(1, self.__ybins+1), xrange(1, self.__zbins+1))

  def __domirror(self):
    for x, y, z in self.binsxyz:
      if y > self.__ybins / 2: continue
      sign = {"symmetric": 1, "antisymmetric": -1}[self.__mirrortype]
      newbincontent = weightedaverage((
        self.__h.GetBinContentError(x, y, z),
        sign * self.__h.GetBinContentError(x, self.__ybins+1-y, z),
      ))

      self.__h.SetBinContent(x, y, z, newbincontent.nominal_value)
      self.__h.SetBinError(x, y, z, newbincontent.std_dev)

      self.__h.SetBinContent(x, self.__ybins+1-y, z, sign*newbincontent.nominal_value)
      self.__h.SetBinError(x, self.__ybins+1-y, z, newbincontent.std_dev)

  def __dofloor(self):
    floor = self.__floor
    if floor.nominal_value <= 0:
      raise ValueError("Invalid floor {} has to be positive.".format(floor.nominal_value))

    if floor.std_dev == 0:
      #use this procedure to estimate the error for floored bins
      maxerrorratio, errortoset = max(
        (self.__h.GetBinError(x, y, z) / self.__h.GetBinContent(x, y, z), self.__h.GetBinError(x, y, z))
          for x in xrange(1, self.__xbins+1)
          for y in xrange(1, self.__ybins+1)
          for z in xrange(1, self.__zbins+1)
        if self.__h.GetBinContent(x, y, z) != 0
      )
      #the reasoning being that if there's a bin with just one entry 2.3 +/- 2.3, then the zero bin could also have 2.3
      #but we can't draw that conclusion from a bin 1000 +/- 5.5

      floor = uncertainties.ufloat(floor.nominal_value, errortoset)

    for x, y, z in self.binsxyz:
      if self.__h.GetBinContent(x, y, z) <= floor.nominal_value:
        self.__h.SetBinContent(x, y, z, floor.nominal_value)
        self.__h.SetBinError(x, y, z, floor.std_dev)

  def makefinaltemplate(self):
    for component in self.__templatecomponents:
      component.lock()

    flooredbins = []

    for x, y, z in self.binsxyz:
      bincontent = []
      for component in self.__templatecomponents:
        bincontent.append(component.GetBinContentError(x, y, z))
      finalbincontent = weightedaverage(bincontent) * self.__scaleby
      if floor is not None and finalbincontent.nominal_value <= 0:
        finalbincontent = floor
        flooredbins.append((x, y, z))
      self.__h.SetBinContent(x, y, z, finalbincontent.nominal_value)
      self.__h.SetBinError(finalbincontent.std_dev)

    if self.__mirrortype is not None: self.__domirror()
    if self.__floor is not None: self.__dofloor()
