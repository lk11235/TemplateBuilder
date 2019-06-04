from __future__ import print_function

import abc, copy, itertools, textwrap

import numpy as np
from scipy import optimize
from uncertainties import ufloat

from cuttingplanemethod import cuttingplanemethod1dquadratic, cuttingplanemethod1dquartic, cuttingplanemethod3dquadratic, cuttingplanemethod4dquadratic, cuttingplanemethod4dquartic, cuttingplanemethod4dquartic_4thvariablequadratic, cuttingplanemethod4dquartic_4thvariablezerobeyondquadratic, cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic
from moremath import kspoissongaussian, weightedaverage
from optimizeresult import OptimizeResult

def ConstrainedTemplates(constrainttype, *args, **kwargs):
  return {
    "unconstrained": OneTemplate,
    "oneparameterggH": OneParameterggH,
    "oneparameterVVH": OneParameterVVH,
    "threeparameterggH": ThreeParameterggH,
    "fourparameterggH": FourParameterggH,
    "fourparameterVVH": FourParameterVVH,
    "fourparameterWWH": FourParameterWWH,
  }[constrainttype](*args, **kwargs)

class ConstrainedTemplatesBase(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, templates, logfile=None):
    self.__templates = templates
    if len(templates) != self.ntemplates:
      raise ValueError("Wrong number of templates ({}) for {}, should be {}".format(len(templates), type(self).__name__, self.ntemplates))
    self.__logfile = logfile

  @property
  def templates(self): 
    return self.__templates

  @property
  def ntemplates(self): return len(self.templatenames)

  @abc.abstractproperty
  def templatenames(self): "can be a class member, names of the templates in order (e.g. SM, int, BSM)"

  @property
  def binsxyz(self):
    for binxyz in itertools.izip(*(t.binsxyz for t in self.templates)):
      binxyz = set(binxyz)
      if len(binxyz) != 1:
        raise ValueError("Templates have inconsistent binning")
      yield binxyz.pop()

  @property
  def xbins(self):
    result = {t.xbins for t in self.templates}
    assert len(result) == 1
    return result.pop()

  @property
  def ybins(self):
    result = {t.ybins for t in self.templates}
    assert len(result) == 1
    return result.pop()

  @property
  def zbins(self):
    result = {t.zbins for t in self.templates}
    assert len(result) == 1
    return result.pop()

  def getcomponentbincontents(self, x, y, z):
      bincontents = []

      for _ in self.templates:
        thisonescontent = {}
        bincontents.append(thisonescontent)
        for component in _.templatecomponents:
          thisonescontent[component.name.replace(_.name+"_", "")] = component.GetBinContentError(x, y, z)

      return bincontents

  def getcomponentbincontentsabs(self, x, y, z):
      bincontents = []

      for _ in self.templates:
        thisonescontent = {}
        bincontents.append(thisonescontent)
        for component in _.templatecomponents:
          thisonescontent[component.name.replace(_.name+"_", "")] = component.GetBinContentErrorAbs(x, y, z)

      return bincontents

  def write(self, thing):
    print(thing)
    if self.__logfile is not None:
      self.__logfile.write(thing+"\n")

  def makefinaltemplates(self, printbins, printallbins, binsortkey=None):
    if all(_.alreadyexists for _ in self.templates):
      for _ in self.templates:
        for component in _.templatecomponents:
          component.lock()
      return
    assert not any(_.alreadyexists for _ in self.templates)

    printbins = tuple(tuple(_) for _ in printbins)
    assert all(len(_) == 3 for _ in printbins)
    self.write("Making the final templates:")
    for name, _ in itertools.izip(self.templatenames, self.templates):
      self.write("  {:>10}: {:40} {:45}".format(name, _.printprefix, _.name))
    self.write("from individual templates with integrals:")

    printedbins = []
    warnings = []

    for _ in self.templates:
      for component in _.templatecomponents:
        component.lock()
        self.write("  {:45} {:10.3e}".format(component.name, component.integral))

    for x, y, z in sorted(self.binsxyz, key=binsortkey):
      bincontents = self.getcomponentbincontents(x, y, z)
      bincontentsabs = self.getcomponentbincontentsabs(x, y, z)

      assert len({frozenset(_) for _ in bincontents}) == 1  #they should all have the same keys

      try:
        finalbincontents, printmessage, warning = self.computefinalbincontents(bincontents, bincontentsabs)
      except BaseException as e:
        print("Error when finding content for bin", x, y, z)
        if hasattr(e, "thingtoprint"): print(e.thingtoprint)
        raise

      for t, content in itertools.izip(self.templates, finalbincontents):
        t.SetBinContentError(x, y, z, content)

      printmessage = "  {:3d} {:3d} {:3d}:\n".format(x, y, z) + printmessage
      if (x, y, z) in printbins:
        printedbins.append(printmessage)
      if printallbins:
        self.write(printmessage)
      else:
        print("  {:3d} {:3d} {:3d}".format(x, y, z))

      if warning or printallbins:
        if isinstance(warning, basestring):
          warning = [warning]
        else:
          warning = list(warning)
        warnings.append(
          "\n      ".join(
            ["  {:3d} {:3d} {:3d}: (pure bin contents: {})".format(x, y, z, ", ".join(str(_) for _ in self.purebincontents(x, y, z)))]
            +warning
          )
        )

    if printedbins:
      self.write("")
      self.write("Bins you requested to print:")
      for _ in printedbins: self.write(_)

    if warnings:
      self.write("")
      self.write("Warnings:")
      for _ in warnings: self.write(_)

    for _ in self.templates:
      _.finalize()

    self.write("")
    self.write("final integrals:")
    for name, t in itertools.izip(self.templatenames, self.templates):
      self.write("  {:>10} = {:10.3e}".format(name, t.integral))
    self.write("")


  @abc.abstractmethod
  def computefinalbincontents(self, bincontents, bincontentsabs): pass

  def applymirrortoarray(self, array):
    if len(self.templates) != len(array):
      raise ValueError("array should have length {}".format(len(self.templates)))
    try:
      return np.array([
        {"symmetric": 1, "antisymmetric": -1}[t.mirrortype] * s
           if t.scaleby != 0
           else 0 * s
        for t, s in itertools.izip(self.templates, array)
      ])
    except KeyError:
      raise ValueError("Not all templates have mirror: " + ", ".join(str(t.mirrortype) for t in self.templates))

  def findoutliers(self, bincontent, bincontentabs, debugprint=False):
    bincontent = bincontent.copy()
    bincontentabs = bincontentabs.copy()

    relativeerror = {name: contentabs.s / contentabs.n if contentabs.n else float("inf") for name, contentabs in bincontentabs.iteritems()}
    outliers = {}

    for name in bincontent:
      if debugprint: print(name, bincontent[name], relativeerror[name], bincontentabs[name])
      if debugprint: print("still here!")
      errortoset = None
      for othername in bincontent:
        #look at the other names that have bigger errors but comparable relative errors
        if bincontentabs[othername].s <= bincontentabs[name].s: continue
        if bincontentabs[name].s == 0: continue
        if debugprint: print("here with", othername)
        if relativeerror[othername] <= relativeerror[name] * (
          (1 + 1.5 * np.log10(bincontentabs[othername].s / bincontentabs[name].s) * kspoissongaussian(1/relativeerror[name]**2))
        ):
          if debugprint: print("here 2 with", othername)
          if errortoset is None: errortoset = 0
          errortoset = max(errortoset, bincontentabs[othername].s)
      if errortoset is not None:
        outliers[name] = ufloat(bincontent[name].n, errortoset)

    return outliers

  def purebincontents(self, x, y, z):
    for t in self.puretemplates:
      yield t.GetBinContentError(x, y, z)

  @property
  def puretemplates(self):
    return [self.templates[i] for i in self.pureindices]

class OneTemplate(ConstrainedTemplatesBase):
  templatenames = "",
  pureindices = 0,

  def computefinalbincontents(self, bincontents, bincontentsabs):
    bincontent = bincontents[0].copy()
    originalcontent = copy.deepcopy(bincontent)
    bincontentabs = bincontentsabs[0].copy()
    nbincontents = len(bincontent)

    warning = []

    outliers = self.findoutliers(bincontent, bincontentabs)
    bincontent.update(outliers)
    if outliers:
      warning.append("some errors have been inflated: " + ", ".join(sorted(outliers)))

    if all(_.n == _.s == 0 for _ in bincontent.itervalues()):  #special case, empty bin
      finalbincontent = bincontent.values()[0]
    else:                                                      #normal case
      finalbincontent = weightedaverage(bincontent.itervalues())
    thingtoprint = ""
    fmt1 = "      {:<%d} {:10.3e}" % max(len(name) for name in bincontent)
    fmt2 = fmt1 + " (was +/-{:10.3e})"
    fmt3 = fmt1 + "                     (sum(abs(wt)) {:10.3e})"
    fmt4 = fmt2 + " (sum(abs(wt)) {:10.3e})"
    for i, (name, content) in enumerate(bincontent.iteritems()):
      fmt = {
        (True, True): fmt1,
        (False, True): fmt2,
        (True, False): fmt3,
        (False, False): fmt4,
      }[content.n == originalcontent[name].n and content.s == originalcontent[name].s, i in self.pureindices]
      fmtargs = [name, content]
      if fmt in (fmt2, fmt4): fmtargs.append(originalcontent[name].s)
      if fmt in (fmt3, fmt4): fmtargs.append(bincontentabs[name].n)
      thingtoprint += "\n"+fmt.format(*fmtargs)

    thingtoprint += "\n"+fmt1.format("final", finalbincontent)

    return [finalbincontent], thingtoprint, warning

class ConstrainedTemplatesWithFit(ConstrainedTemplatesBase):
  def computefinalbincontents(self, bincontents, bincontentsabs):
    warning = []

    bincontents = copy.deepcopy(bincontents)
    bincontentsabs = copy.deepcopy(bincontentsabs)
    originalbincontents = copy.deepcopy(bincontents)
    nbincontents = len(bincontents[0])

    #Each template component produces a 3D probability distribution in (SM, int, BSM)
    #FIXME: include correlations and don't approximate as Gaussian

    x0 = [[] for t in self.templates]
    sigma = [[] for t in self.templates]

    for bincontent, bincontentabs, t in itertools.izip(bincontents, bincontentsabs, self.templates):
      outliers = self.findoutliers(bincontent, bincontentabs)
      bincontent.update(outliers)
      if outliers:
        warning.append("some errors have been inflated for "+t.name+": "+", ".join(sorted(outliers)))

    for name in bincontents[0]:
      for thisonescontent, thisx0, thissigma in itertools.izip(bincontents, x0, sigma):
        thisx0.append(thisonescontent[name].n)
        thissigma.append(thisonescontent[name].s)

    x0 = np.array(x0)
    sigma = np.array(sigma)

    thingtoprint = ""
    fmt1 = "      {:<%d} {:10.3e}" % max(len(name) for name in bincontent)
    fmt2 = fmt1 + " (was +/-{:10.3e})"
    fmt3 = fmt1 + "                     (sum(abs(wt)) {:10.3e})"
    fmt4 = fmt2 + " (sum(abs(wt)) {:10.3e})"
    for i, (t, thisonescontent, originalcontent, bincontentabs) in enumerate(itertools.izip(self.templates, bincontents, originalbincontents, bincontentsabs)):
      thingtoprint += "\n"+t.name+":"
      for name, content in sorted(thisonescontent.iteritems()):
        fmt = {
          (True, True): fmt1,
          (False, True): fmt2,
          (True, False): fmt3,
          (False, False): fmt4,
        }[content.n == originalcontent[name].n and content.s == originalcontent[name].s, i in self.pureindices]
        fmtargs = [name, content]
        if fmt in (fmt2, fmt4): fmtargs.append(originalcontent[name].s)
        if fmt in (fmt3, fmt4): fmtargs.append(bincontentabs[name].n)
        thingtoprint += "\n"+fmt.format(*fmtargs)

    if not np.any(np.nonzero(x0)):
      finalbincontents = np.array([0]*self.ntemplates)
      fitprintmessage = "all templates have zero content for this bin"
    elif all(len(_[np.nonzero(_)]) == 1 for _ in x0):
      finalbincontents = [
        _[np.nonzero(_)][0] for _ in x0
      ]
      fitprintmessage = "only one reweighted sample has events in this bin, using that directly"
    else:
      cachekey = tuple(tuple(_) for _ in x0), tuple(tuple(_) for _ in sigma)
      if all(t.mirrortype for t in self.templates):
        mirroredx0 = self.applymirrortoarray(x0)
        mirroredcachekey = tuple(tuple(_) for _ in mirroredx0), tuple(tuple(_) for _ in sigma)
      try:
        if cachekey not in self.__fitresultscache:
          fitresult = self.__fitresultscache[cachekey] = self.docuttingplanes(
            x0,
            sigma,
          )
          if all(t.mirrortype for t in self.templates):
            self.__fitresultscache[mirroredcachekey] = OptimizeResult(
              x=self.applymirrortoarray(fitresult.x),
              fun=fitresult.fun,
              message="(mirrored) "+fitresult.message,
              status=fitresult.status,
              nit=fitresult.nit,
              maxcv=fitresult.maxcv
            )
        fitresult = self.__fitresultscache[cachekey]

        if fitresult.maxcv:
          raise ValueError("Fit failed with constraint violation\n\n{}".format(fitresult))

      except BaseException as e:
        e.thingtoprint = thingtoprint
        raise

      finalbincontents = fitresult.x

      if fitresult.nit == 1:
        fitprintmessage = "global minimum already satisfies constraint"
      else:
        fitprintmessage = str(fitresult)
        warning.append("fit converged in {0.nit} with NLL = {0.fun}".format(fitresult))
        warning.append(fitresult.message)

    thingtoprint += "\n\n"+str(fitprintmessage)+"\n"
    for name, content in itertools.izip(self.templatenames, finalbincontents):
      thingtoprint += "\n"+fmt1.format("final "+name, content)

    return finalbincontents, thingtoprint.lstrip("\n"), warning

  def __init__(self, *args, **kwargs):
    super(ConstrainedTemplatesWithFit, self).__init__(*args, **kwargs)
    self.__fitresultscache = {}

  @abc.abstractproperty
  def pureindices(self): "can be a class member"
  @abc.abstractmethod
  def cuttingplanefunction(self, x0, sigma, maxfractionaladjustment): "can be static"
  @abc.abstractproperty
  def cuttingplanehaspermutations(self):
    """
    can be a class member
    say if the cutting plane function has a usepermutations kwarg
    """

  def docuttingplanes(self, x0, sigma, maxfractionaladjustment=1e-6, maxiter=100):
    try:
      result = self.cuttingplanefunction(x0, sigma, maxfractionaladjustment=maxfractionaladjustment, maxiter=maxiter)
      if self.cuttingplanehaspermutations and result.status >= 3: raise Exception
      return result
    except Exception as e:
      if self.cuttingplanehaspermutations:
        return self.cuttingplanefunction(x0, sigma, maxfractionaladjustment=maxfractionaladjustment, maxiter=maxiter, usepermutations=True)
      raise

class OneParameterggH(ConstrainedTemplatesWithFit):
  templatenames = "SM", "int", "BSM"
  pureindices = 0, 2
  cuttingplanefunction = staticmethod(cuttingplanemethod1dquadratic)
  cuttingplanehaspermutations = False

class OneParameterVVH(ConstrainedTemplatesWithFit):
  templatenames = "SM", "g13gi1", "g12gi2", "g11gi3", "BSM"
  pureindices = 0, 4
  cuttingplanefunction = staticmethod(cuttingplanemethod1dquartic)
  cuttingplanehaspermutations = False

class ThreeParameterggH(ConstrainedTemplatesWithFit):
  templatenames = (
    "SM", "g11gj1", "g11gk1", "g11gl1",
    "j",  "gj1gk1", "gj1gl1",
    "k",  "gk1gl1",
    "l",
  )
  pureindices = 0, 4, 7, 9
  cuttingplanefunction = staticmethod(cuttingplanemethod3dquadratic)
  cuttingplanehaspermutations = True

class FourParameterggH(ConstrainedTemplatesWithFit):
  templatenames = (
    "SM", "g11gi1", "g11gj1", "g11gk1", "g11gl1",
    "i",  "gi1gj1", "gi1gk1", "gi1gl1",
    "j",  "gj1gk1", "gj1gl1",
    "k",  "gk1gl1",
    "l",
  )
  pureindices = 0, 5, 9, 12, 14
  cuttingplanefunction = staticmethod(cuttingplanemethod4dquadratic)
  cuttingplanehaspermutations = True

class FourParameterVVH(ConstrainedTemplatesWithFit):
  templatenames = (
    "SM",
    "g13gi1", "g13gj1",    "g13gk1",       "g13gl1",

    "g12gi2", "g12gi1gj1", "g12gi1gk1",    "g12gi1gl1",
              "g12gj2",    "g12gj1gk1",    "g12gj1gl1",
                           "g12gk2",       "g12gk1gl1",
                                           "g12gl2",

    "g11gi3", "g11gi2gj1", "g11gi2gk1",    "g11gi2gl1",
              "g11gi1gj2", "g11gi1gj1gk1", "g11gi1gj1gl1",
                           "g11gi1gk2",    "g11gi1gk1gl1",
                                           "g11gi1gl2",
              "g11gj3",    "g11gj2gk1",    "g11gj2gl1",
                           "g11gj1gk2",    "g11gj1gk1gl1",
                                           "g11gj1gl2",
                           "g11gk3",       "g11gk2gl1",
                                           "g11gk1gl2",
                                           "g11gl3",

    "i",
              "gi3gj1",    "gi3gk1",       "gi3gl1",

              "gi2gj2",    "gi2gj1gk1",    "gi2gj1gl1",
                           "gi2gk2",       "gi2gk1gl1",
                                           "gi2gl2",

              "gi1gj3",    "gi1gj2gk1",    "gi1gj2gl1",
                           "gi1gj1gk2",    "gi1gj1gk1gl1",
                                           "gi1gj1gl2",
                           "gi1gk3",       "gi1gk2gl1",
                                           "gi1gk1gl2",
                                           "gi1gl3",

    "j",
                           "gj3gk1",       "gj3gl1",

                           "gj2gk2",       "gj2gk1gl1",
                                           "gj2gl2",

                           "gj1gk3",       "gj1gk2gl1",
                                           "gj1gk1gl2",
                                           "gj1gl3",

    "k",
                                           "gk3gl1",

                                           "gk2gl2",

                                           "gk1gl3",

    "l",

  )
  pureindices = 0, 35, 55, 65, 69
  def cuttingplanefunction(self, x0, sigma, *args, **kwargs):
    if np.all(x0[self.gZ3indices,:] == 0): #this happens for VBF when there are no reweighted ZZ fusion events in the bin
      return cuttingplanemethod4dquartic_4thvariablezerobeyondquadratic(x0, sigma, *args, **kwargs)

    elif max(
      weightedaverage(
        ufloat(x0ij, sigmaij)
        for x0ij, sigmaij in itertools.izip(x0[i], sigma[i])
      )
      for i in range(self.ntemplates)
      if i in self.gZ3indices and i in self.pureindices
    ) / min(
      weightedaverage(
        ufloat(x0ij, sigmaij)
        for x0ij, sigmaij in itertools.izip(x0[i], sigma[i])
      )
      for i in range(self.ntemplates)
      if i not in self.gZ3indices and i in self.pureindices
    ) < 1e-3:
      return cuttingplanemethod4dquartic_4thvariablesmallbeyondquadratic(x0, sigma, *args, **kwargs)

    else:
      return cuttingplanemethod4dquartic(x0, sigma, *args, **kwargs)
  gZ3indices = tuple(i for i, _ in enumerate(templatenames) if "gl3" in _ or _ == "l")
  cuttingplanehaspermutations = True

class FourParameterWWH(ConstrainedTemplatesWithFit):
  #https://stackoverflow.com/q/13905741/5228524
  templatenames = tuple(name for i, name in enumerate(FourParameterVVH.templatenames) if i not in FourParameterVVH.gZ3indices)
  pureindices = tuple(
    index - sum(1 for i in FourParameterVVH.gZ3indices if i < index)
    for index in FourParameterVVH.pureindices
    if index not in FourParameterVVH.gZ3indices
  )
  cuttingplanefunction = staticmethod(cuttingplanemethod4dquartic_4thvariablequadratic)
  cuttingplanehaspermutations = True
