from __future__ import print_function

import abc, copy, itertools, textwrap

try:
  import autograd
except ImportError:
  autograd = None
  import numpy as np
else:
  import autograd.numpy as np

from scipy import optimize
if hasattr(optimize, "NonlinearConstraint"):
  hasscipy = True
else:
  hasscipy = False

from uncertainties import ufloat

from moremath import kspoissongaussian, minimizequartic, minimizequartic4d, weightedaverage


def ConstrainedTemplates(constrainttype, *args, **kwargs):
  return {
    "unconstrained": OneTemplate,
    "oneparameterggH": OneParameterggH,
    "oneparameterVVH": OneParameterVVH,
    "fourparameterggH": FourParameterggH,
    "fourparameterVVH": FourParameterVVH,
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
      except:
        print("Error when finding content for bin", x, y, z)
        raise

      for t, content in itertools.izip(self.templates, finalbincontents):
        t.SetBinContentError(x, y, z, content)

      printmessage = "  {:3d} {:3d} {:3d}:\n".format(x, y, z) + printmessage
      if (x, y, z) in printbins:
        printedbins.append(printmessage)
      if printallbins:
        self.write(printmessage)

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
    return np.array([
      {"symmetric": 1, "antisymmetric": -1}[t.mirrortype] * s
      for t, s in itertools.izip(self.templates, array)
    ])

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
        if bincontentabs[othername].s < bincontentabs[name].s: continue
        if debugprint: print("here with", othername)
        if relativeerror[othername] <= relativeerror[name] * (
          (1+1.5*kspoissongaussian(1/relativeerror[name]**2))
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
  pureindices = 0, 2

  def computefinalbincontents(self, bincontents, bincontentsabs):
    bincontent = bincontents[0].copy()
    originalbincontent = copy.deepcopy(bincontent)
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
    fmt4 = fmt2 + " (sum of abs weights {:10.3e})"
    for i, (name, content) in enumerate(bincontent.iteritems()):
      fmt = {
        (True, True): fmt1,
        (False, True): fmt2,
        (True, False): fmt3,
        (False, False): fmt4,
      }[content.n == originalcontent[name].n and content.s == originalcontent[name].s, i in self.pureindices]
      fmtargs = [name, content]
      if fmt in (fmt2, fmt4): fmtargs.append(originalcontent[name])
      if fmt in (fmt3, fmt4): fmtargs.append(bincontentabs[name])
      thingtoprint += "\n"+fmt.format(*fmtargs)

    thingtoprint += "\n"+fmt.format("final", finalbincontent)

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
      for name in list(bincontent):
        if (
          bincontent[name].n == 0    #0 content - maginfy error to the maximum error
          or (                            #large relative error and this is the only nonzero one - same
            bincontent[name].s == abs(bincontent[name].n)
            and all(othercontent.n == 0 for othername, othercontent in bincontent.iteritems() if othername != name)
          )
        ):
          bincontent[name] = ufloat(bincontent[name].n, max(othercontent.s for othercontent in bincontent.itervalues()))

      outliers = self.findoutliers(bincontent, bincontentabs)
      bincontent.update(outliers)
      if outliers:
        warning.append("some errors have been inflated for "+t.name+": "+", ".join(sorted(outliers)))

    for name in bincontents[0]:
      for thisonescontent, thisx0, thissigma in itertools.izip(bincontents, x0, sigma):
        thisx0.append(thisonescontent[name].n)
        thissigma.append(thisonescontent[name].s)

    x0 = [np.array(_) for _ in x0]
    sigma = [np.array(_) for _ in sigma]

    startpoint = np.array([weightedaverage(_.itervalues()).n for _ in bincontents])
    for i in self.pureindices:
      if startpoint[i] == 0:
        if all(startpoint[i] <= np.finfo(float).eps for i in self.pureindices):
          startpoint[i] = np.finfo(float).eps
        else:
          startpoint[i] = min(startpoint[i] for i in self.pureindices if startpoint[i] > np.finfo(float).eps)

    constraint = self.constraint

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

    mirroredstartpoint = self.applymirrortoarray(startpoint)
    if tuple(startpoint) not in self.__constraintatstartcache:
      constraintatstart = constraint(startpoint)
      self.__constraintatstartcache[tuple(startpoint)] = self.__constraintatstartcache[tuple(mirroredstartpoint)] = constraintatstart
    else:
      constraintatstart = self.__constraintatstartcache[tuple(startpoint)]

    if np.all(self.constraintmin <= constraintatstart) and np.all(constraintatstart <= self.constraintmax):
      fitprintmessage = "no need for a fit - average already satisfies the constraint"
      finalbincontents = startpoint
    else:
      fitprintmessage = textwrap.dedent("""
        multiply by {:.0e} for numerical stability
        weighted averages:
        {}
        adjust to constraint --> fit starting from:
        {} (NLL = {})

        bounds:
        {}

        result:
        {}
      """)

      try:
        multiply = 10 ** -min(np.floor(np.log10(abs(startpoint[np.nonzero(startpoint)]))))
        startpoint *= multiply
        fitstartpoint = self.adjuststartpoint(startpoint, constraint, self.constraintmin, self.constraintmax)
      except:
        print("Error (probably a math error) involving startpoint:")
        print(startpoint)
        print()
        raise

      negativeloglikelihood = self.makeNLL(x0, sigma, nbincontents, multiply=multiply)

      if self.minimizemethod == "trust-constr":
        nlljacobian = autograd.jacobian(negativeloglikelihood)
        nllhessian = autograd.hessian(negativeloglikelihood)

        bounds = self.bounds(fitstartpoint, multiply)

        constraintjacobian = autograd.jacobian(constraint)
        constrainthessianv = autograd.linear_combination_of_hessians(constraint)
        constraints = [optimize.NonlinearConstraint(constraint, self.constraintmin, self.constraintmax, constraintjacobian, constrainthessianv)]

      elif self.minimizemethod == "COBYLA":
        nlljacobian = nllhessian = bounds = None
        constraints = [
          {
            "type": "ineq",
            "fun": constraint,
          }
        ]
      else:
        assert False, self.minimizemethod

      try:
        if tuple(startpoint) not in self.__fitresultscache:
          #use startpoint as the key, not fitstartpoint,
          #because fitstartpoint has more numerical operations on it
          #and therefore leads to error
          fitresult = self.__fitresultscache[tuple(startpoint)] = optimize.minimize(
            negativeloglikelihood,
            fitstartpoint,
            method=self.minimizemethod,
            jac=nlljacobian,
            hess=nllhessian,
            constraints=constraints,
            bounds=bounds,
            options = {},
          )
          if fitresult.fun > negativeloglikelihood(fitstartpoint):
            fitresult = self.__fitresultscache[tuple(startpoint)] = optimize.OptimizeResult(
              message=fitresult.message + "\nx = {0.x}, fun = {0.fun}, but the starting point was better, using that instead".format(fitresult),
              x=fitstartpoint,
              fun=negativeloglikelihood(fitstartpoint),
              status=fitresult.status * -1,
            )
          if all(t.mirrortype for t in self.templates):
            self.__fitresultscache[tuple(mirroredstartpoint)] = optimize.OptimizeResult(
              x=self.applymirrortoarray(fitresult.x),
              fun=fitresult.fun,
              message="(mirrored) "+fitresult.message,
              status=fitresult.status,
            )
        fitresult = self.__fitresultscache[tuple(startpoint)]

      except:
        print(thingtoprint+"\n\n"+fitprintmessage.format(multiply, startpoint, fitstartpoint, negativeloglikelihood(fitstartpoint), bounds, ""))
        raise

      fitprintmessage = fitprintmessage.format(multiply, startpoint, fitstartpoint, negativeloglikelihood(fitstartpoint), bounds, fitresult).strip()

      finalbincontents = fitresult.x / multiply

      warning.append("fit converged with NLL = {}".format(fitresult.fun))
      if fitresult.status not in (1, 2): warning.append(fitresult.message)

    thingtoprint += "\n\n"+str(fitprintmessage)+"\n"
    for name, content in itertools.izip(self.templatenames, finalbincontents):
      thingtoprint += "\n"+fmt1.format("final "+name, content)

    return finalbincontents, thingtoprint.lstrip("\n"), warning

  def bounds(self, fitstartpoint, multiply):
    """
    most lenient possible bounds
    override this to get better results!
    """
    return optimize.Bounds(
      np.array([np.finfo(float).eps if i in self.pureindices else -np.inf for i in xrange(self.ntemplates)]) * multiply,
      np.array([np.inf for i in xrange(self.ntemplates)]) * multiply,
      keep_feasible=True,
    )

  def adjuststartpoint(self, startpoint, constraint, constraintmin, constraintmax):
    result = np.copy(startpoint)

    for constraintidx in xrange(len(constraint(startpoint))):
      startpoint = result

      if (constraintmin < constraint(startpoint))[constraintidx]: continue

      #we want to adjust the start point in a reasonable way so that it fills the constraint
      #the simple way to do this is to increase the pure components by some factor
      increasepureindices = np.array([1 if i in self.pureindices else 0 for i, _ in enumerate(startpoint)])
      def functiontosolvefor0(x):
        print(x, constraint(startpoint * (increasepureindices*x + 1))[constraintidx] - constraintmin)
        return constraint(startpoint * (increasepureindices*x + 1))[constraintidx] - constraintmin
      assert functiontosolvefor0(0) < 0

      for x in itertools.count(0):
        if functiontosolvefor0(x) > 0:
          break

      fprime = fprime2 = None
      if self.candifferentiateconstraint:
        fprime = autograd.grad(functiontosolvefor0)
        fprime2 = autograd.grad(fprime)
      increaseby = optimize.brentq(functiontosolvefor0, x-1, x)

      for i in xrange(10000):
        result = startpoint * (increasepureindices*increaseby + 1)
        if (constraintmin < constraint(result))[constraintidx]:
          break
        increaseby *= 1.0000001
        if i > 5000: print(i, increaseby, result, constraint(result))
      else:
        raise RuntimeError("increasing didn't work")

    assert np.all(constraintmin < constraint(result)), constraint(result)
    assert np.all(constraint(result) < constraintmax), constraint(result)

    return result

  def __init__(self, *args, **kwargs):
    super(ConstrainedTemplatesWithFit, self).__init__(*args, **kwargs)
    if autograd is None:
      raise ImportError("To use "+type(self).__name__+", please install autograd.")
    if not hasscipy:
      raise ImportError("To use "+type(self).__name__+", please install a newer scipy.")
    self.__fitresultscache = {}
    self.__constraintatstartcache = {}

  def makeNLL(self, x0, sigma, nbincontents, multiply):
    def negativeloglikelihood(x):
      return sum(
        ((x[j]/multiply - x0[j][i]) / sigma[j][i]) ** 2
        for i in xrange(nbincontents)
        for j in xrange(self.ntemplates)
      )
    return negativeloglikelihood

  @abc.abstractproperty
  def minimizemethod(self): "can be a class member"
  @abc.abstractproperty
  def candifferentiateconstraint(self): "can be a class member"
  @abc.abstractmethod
  def constraint(self, x): "can be static"
  @abc.abstractproperty
  def constraintmin(self): "can be a class member"
  @abc.abstractproperty
  def constraintmax(self): "can be a class member"

  @abc.abstractproperty
  def pureindices(self): "can be a class member"

class OneParameterggH(ConstrainedTemplatesWithFit):
  templatenames = "SM", "int", "BSM"
  pureindices = 0, 2
  minimizemethod = "trust-constr"
  candifferentiateconstraint = True

  @staticmethod
  def constraint(x):
    #|interference| <= 2*sqrt(SM*BSM)
    #2*sqrt(SM*BSM) - |interference| >= 0
    return np.array([2*(x[0]*x[2])**.5 - abs(x[1])])

  constraintmin = np.finfo(np.float).eps
  constraintmax = np.inf

class OneParameterVVH(ConstrainedTemplatesWithFit):
  templatenames = "SM", "g13gi1", "g12gi2", "g11gi3", "BSM"
  pureindices = 0, 4
  minimizemethod = "trust-constr"
  candifferentiateconstraint = True

  @staticmethod
  def constraint(x):
    return np.array([minimizequartic(x)])

  constraintmin = np.finfo(np.float).eps
  constraintmax = np.inf

  def bounds(self, fitstartpoint, multiply):
    maxevenstartpoint = max(_ for i, _ in enumerate(fitstartpoint) if i in (0, 2, 4))
    return optimize.Bounds(
      np.array([np.finfo(float).eps, -10*maxevenstartpoint, -10*maxevenstartpoint, -10*maxevenstartpoint, np.finfo(float).eps]) * multiply,
      np.array([2*fitstartpoint[0],   10*maxevenstartpoint,  10*maxevenstartpoint,  10*maxevenstartpoint, 2*fitstartpoint[4] ]) * multiply,
      keep_feasible=True,
    )

class FourParameterggH(ConstrainedTemplatesWithFit):
  templatenames = (
    "SM", "g11gi1", "g11gj1", "g11gk1", "g11gl1",
    "i",  "gi1gj1", "gi1gk1", "gi1gl1",
    "j",  "gj1gk1", "gj1gl1",
    "k",  "gk1gl1",
    "l",
  )
  pureindices = 0, 5, 9, 12, 14
  minimizemethod = "trust-constr"
  candifferentiateconstraint = True

  @staticmethod
  def constraint(x):
    return np.array([
      2*(x[0 ]*x[5 ])**.5 - abs(x[1 ]),
      2*(x[0 ]*x[9 ])**.5 - abs(x[2 ]),
      2*(x[0 ]*x[12])**.5 - abs(x[3 ]),
      2*(x[0 ]*x[14])**.5 - abs(x[4 ]),
      2*(x[5 ]*x[9 ])**.5 - abs(x[6 ]),
      2*(x[5 ]*x[12])**.5 - abs(x[7 ]),
      2*(x[5 ]*x[14])**.5 - abs(x[8 ]),
      2*(x[9 ]*x[12])**.5 - abs(x[10]),
      2*(x[9 ]*x[14])**.5 - abs(x[11]),
      2*(x[12]*x[14])**.5 - abs(x[13]),
    ])

  constraintmin = np.finfo(np.float).eps
  constraintmax = np.inf

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
  minimizemethod = "COBYLA"
  candifferentiateconstraint = False

  @staticmethod
  def constraint(x):
    return np.array([minimizequartic4d(x)])

  constraintmin = np.finfo(np.float).eps
  constraintmax = np.inf
