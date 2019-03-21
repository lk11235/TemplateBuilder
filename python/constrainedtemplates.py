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

from moremath import minimizequartic, weightedaverage


def ConstrainedTemplates(constrainttype, templates):
  return {
    "unconstrained": OneTemplate,
    "oneparameterggH": OneParameterggH,
    "oneparameterVVH": OneParameterVVH,
  }[constrainttype](templates)

class ConstrainedTemplatesBase(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, templates):
    self.__templates = templates
    if len(templates) != self.ntemplates:
      raise ValueError("Wrong number of templates ({}) for {}, should be {}".format(len(templates), type(self).__name__, ntemplates))

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

  def makefinaltemplates(self, printbins, printallbins):
    printbins = tuple(tuple(_) for _ in printbins)
    assert all(len(_) == 3 for _ in printbins)
    print
    print "Making the final templates:"
    for name, _ in itertools.izip(self.templatenames, self.templates):
      print "  {:>10}: {:40} {:45}".format(name, _.printprefix, _.name)
    print "from individual templates with integrals:"

    printedbins = []
    warnings = []

    for _ in self.templates:
      for component in _.templatecomponents:
        component.lock()
        print "  {:45} {:10.3e}".format(component.name, component.integral)

    for x, y, z in self.binsxyz:
      bincontents = self.getcomponentbincontents(x, y, z)

      assert len({frozenset(_) for _ in bincontents}) == 1  #they should all have the same keys

      try:
        finalbincontents, printmessage, warning = self.computefinalbincontents(bincontents)
      except:
        print "Error when finding content for bin", x, y, z
        raise

      for t, content in itertools.izip(self.templates, finalbincontents):
        t.SetBinContentError(x, y, z, content)

      printmessage = "  {:3d} {:3d} {:3d}:\n".format(x, y, z) + printmessage
      if (x, y, z) in printbins:
        printedbins.append(printmessage)
      if printallbins:
        print printmessage

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
      print
      print "Bins you requested to print:"
      for _ in printedbins: print _

    if warnings:
      print
      print "Warnings:"
      for _ in warnings: print _

    for _ in self.templates:
      _.finalize()

    print
    print "final integrals:"
    for name, t in itertools.izip(self.templatenames, self.templates):
      print "  {:>10} = {:10.3e}".format(name, t.integral)
    print


  @abc.abstractmethod
  def computefinalbincontents(self, bincontents): pass

  def applymirrortoarray(self, array):
    if len(self.templates) != len(array):
      raise ValueError("array should have length {}".format(len(self.templates)))
    return np.array([
      {"symmetric": 1, "antisymmetric": -1}[t.mirrortype] * s
      for t, s in itertools.izip(self.templates, array)
    ])

  def findoutliers(self, bincontent):
    bincontent = bincontent.copy()

    if all(_.s == 0 for _ in bincontent.itervalues()): return frozenset()

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
          significance = abs(newunbiasedresidual.n) / content.s
          #if there's a 5sigma difference, then this combination of templatecomponents to remove is no good
          if significance > 5: break #out of the loop over remaining names
        else:
          #no remaining unbiased residuals are 5sigma
          #that means this combination of templatecomponents is a candidate to remove
          #if multiple combinations of the same number of templatecomponents fit this criterion,
          #then we pick the one that itself has the biggest normalized residual from the other templatecomponents
          #therefore we store it in significances
          if contentstomayberemove:
            unbiasedresidual = (
              weightedaverage(contentstomayberemove)
              - weightedaverage(othercontent for othername, othercontent in bincontent.iteritems() if othername not in namestomayberemove)
            )
            significances[namestomayberemove] = abs(unbiasedresidual.n) / weightedaverage(contentstomayberemove).s
          else:
            significances[namestomayberemove] = float("inf")

      if significances:
        nameswithmaxsignificance, maxsignificance = max(significances.iteritems(), key=lambda x: x[1])
        return nameswithmaxsignificance
        break

    return frozenset()

  def purebincontents(self, x, y, z):
    for t in self.puretemplates:
      yield t.GetBinContentError(x, y, z)

  @property
  def puretemplates(self):
    return [self.templates[i] for i in self.pureindices]

class OneTemplate(ConstrainedTemplatesBase):
  templatenames = "",
  pureindices = 0, 2

  def computefinalbincontents(self, bincontents):
    bincontent = bincontents[0]
    nbincontents = len(bincontent)

    namestoremove = self.findoutliers(bincontent)

    warning = []
    if namestoremove:
      warning.append("there are outliers: " + ", ".join(sorted(namestoremove)))
    for name in namestoremove:
      del bincontent[name]

    if len(bincontent) < nbincontents / 2.:
      raise RuntimeError("Removed more than half of the bincontents!  Please check.\n" + "\n".join("  {:45} {:10.3e}".format(component.name, component.GetBinContentError(x, y, z)) for component in template.templatecomponents))

    if all(_.n == _.s == 0 for _ in bincontent.itervalues()):  #special case, empty bin
      finalbincontent = bincontent.values()[0]
    else:                                                      #normal case
      finalbincontent = weightedaverage(bincontent.itervalues())

    thingtoprint = ""
    fmt = "      {:<%d} {:10.3e}" % max(len(name) for name in bincontent)
    for name, content in bincontent.iteritems():
      thingtoprint += "\n"+fmt.format(name, content)
    thingtoprint += "\n"+fmt.format("final", finalbincontent)

    return [finalbincontent], thingtoprint, warning

class ConstrainedTemplatesWithFit(ConstrainedTemplatesBase):
  def computefinalbincontents(self, bincontents):
    warning = []

    bincontents = copy.deepcopy(bincontents)
    originalbincontents = copy.deepcopy(bincontents)
    nbincontents = len(bincontents[0])

    #Each template component produces a 3D probability distribution in (SM, int, BSM)
    #FIXME: include correlations and don't approximate as Gaussian

    x0 = [[] for t in self.templates]
    sigma = [[] for t in self.templates]

    for bincontent, t in itertools.izip(bincontents, self.templates):
      for name in list(bincontent):
        if (
          bincontent[name].n == 0    #0 content - maginfy error to the maximum error
          or (                            #large relative error and this is the only nonzero one - same
            bincontent[name].s == abs(bincontent[name].n)
            and all(othercontent.n == 0 for othername, othercontent in bincontent.iteritems() if othername != name)
          )
        ):
          bincontent[name] = ufloat(bincontent[name].n, max(othercontent.s for othercontent in bincontent.itervalues()))

      outliers = self.findoutliers(bincontent)
      if outliers:
        warning.append("there are outliers for "+t.name+": "+", ".join(sorted(outliers)))
      for name in outliers:
        bincontent[name] = ufloat(bincontent[name].n, max(othercontent.s for othercontent in bincontent.itervalues() if othercontent.n != 0))

    for name in bincontents[0]:
      for thisonescontent, thisx0, thissigma in itertools.izip(bincontents, x0, sigma):
        thisx0.append(thisonescontent[name].n)
        thissigma.append(thisonescontent[name].s)

    x0 = [np.array(_) for _ in x0]
    sigma = [np.array(_) for _ in sigma]

    negativeloglikelihood = self.makeNLL(x0, sigma, nbincontents)
    nlljacobian = autograd.jacobian(negativeloglikelihood)
    nllhessian = autograd.hessian(negativeloglikelihood)

    constraint = self.constraint
    constraintjacobian = autograd.jacobian(constraint)
    constrainthessianv = autograd.linear_combination_of_hessians(constraint)

    nonlinearconstraint = optimize.NonlinearConstraint(constraint, self.constraintmin, self.constraintmax, constraintjacobian, constrainthessianv)

    startpoint = np.array([weightedaverage(_.itervalues()).n for _ in bincontents])
    for i in self.pureindices:
      if startpoint[i] == 0: startpoint[i] = np.finfo(np.float).eps

    thingtoprint = ""
    fmt = "      {:<%d} {:10.3e}" % max(len(name) for name in itertools.chain(*bincontents))
    fmt2 = fmt + " (originally {:10.3e})"
    for t, thisonescontent, originalcontent in itertools.izip(self.templates, bincontents, originalbincontents):
      thingtoprint += "\n"+t.name+":"
      for name, content in sorted(thisonescontent.iteritems()):
        if content.n == originalcontent[name].n and content.s == originalcontent[name].s:
          thingtoprint += "\n"+fmt.format(name, content)
        else:
          thingtoprint += "\n"+fmt2.format(name, content, originalcontent[name])

    if self.constraintmin <= constraint(startpoint) <= self.constraintmax:
      fitprintmessage = "no need for a fit - average already satisfies the constraint"
      finalbincontents = startpoint
    else:
      fitprintmessage = textwrap.dedent("""
        weighted averages:
        {}
        adjust to constraint --> fit starting from:
        {} (NLL = {})

        bounds:
        {}

        result:
        {}
      """)

      fitstartpoint = self.adjuststartpoint(startpoint, constraint)

      bounds = self.bounds(fitstartpoint)

      try:
        if tuple(startpoint) not in self.__fitresultscache:
          #use startpoint as the key, not fitstartpoint,
          #because fitstartpoint has more numerical operations on it
          #and therefore leads to error
          fitresult = self.__fitresultscache[tuple(startpoint)] = optimize.minimize(
            negativeloglikelihood,
            fitstartpoint,
            method='trust-constr',
            jac=nlljacobian,
            hess=nllhessian,
            constraints=[nonlinearconstraint],
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
            mirroredstartpoint = self.applymirrortoarray(startpoint)
            self.__fitresultscache[tuple(mirroredstartpoint)] = optimize.OptimizeResult(
              x=self.applymirrortoarray(fitresult.x),
              fun=fitresult.fun,
              message="(mirrored) "+fitresult.message,
              status=fitresult.status,
            )
        fitresult = self.__fitresultscache[tuple(startpoint)]

      except:
        print thingtoprint+"\n\n"+fitprintmessage.format(startpoint, fitstartpoint, negativeloglikelihood(fitstartpoint), bounds, "")
        raise

      fitprintmessage = fitprintmessage.format(startpoint, fitstartpoint, negativeloglikelihood(fitstartpoint), bounds, fitresult).strip()

      finalbincontents = fitresult.x

      warning.append("fit converged with NLL = {}".format(fitresult.fun))
      if fitresult.status not in (1, 2): warning.append(fitresult.message)

    thingtoprint += "\n\n"+str(fitprintmessage)+"\n"
    for name, content in itertools.izip(self.templatenames, finalbincontents):
      thingtoprint += "\n"+fmt.format("final "+name, content)

    return finalbincontents, thingtoprint.lstrip("\n"), warning

  def bounds(self, fitstartpoint):
    """
    most lenient possible bounds
    override this to get better results!
    """
    return optimize.Bounds(
      [np.finfo(float).eps if i in self.pureindices else -np.inf for i in xrange(self.ntemplates)],
      [np.inf for i in xrange(self.ntemplates)],
      keep_feasible=True,
    )

  def adjuststartpoint(self, startpoint, constraint):
    assert len(constraint(startpoint)) == 1
    assert constraint(startpoint) < 0
    #we want to adjust the start point in a reasonable way so that it fills the constraint
    #the simple way to do this is to increase the pure components by some factor
    increasepureindices = np.array([1 if i in self.pureindices else 0 for i, _ in enumerate(startpoint)])
    def functiontosolvefor0(x):
      return constraint(startpoint * (increasepureindices*x + 1))
    assert functiontosolvefor0(0) < 0

    for x in itertools.count(0):
      if functiontosolvefor0(x) > 0:
        break

    increaseby = optimize.newton(functiontosolvefor0, x-0.5, fprime=autograd.grad(functiontosolvefor0), fprime2=autograd.grad(autograd.grad(functiontosolvefor0)))

    for i in xrange(10000):
      result = startpoint * (increasepureindices*increaseby + 1)
      if constraint(result) > 0:
        return result
      increaseby *= 1.0000001
      if i > 5000: print i, increaseby, result, constraint(result)

    raise RuntimeError("increasing didn't work")

  def __init__(self, *args, **kwargs):
    super(ConstrainedTemplatesWithFit, self).__init__(*args, **kwargs)
    if autograd is None:
      raise ImportError("To use "+type(self).__name__+", please install autograd.")
    if not hasscipy:
      raise ImportError("To use "+type(self).__name__+", please install a newer scipy.")
    self.__fitresultscache = {}

  @abc.abstractmethod
  def makeNLL(self, x0, sigma, nbincontents): pass

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

  @staticmethod
  def constraint(x):
    #|interference| <= 2*sqrt(SM*BSM)
    #2*sqrt(SM*BSM) - |interference| >= 0
    return np.array([2*(x[0]*x[2])**.5 - abs(x[1])])

  constraintmin = np.finfo(np.float).eps
  constraintmax = np.inf

  def makeNLL(self, x0, sigma, nbincontents):
    def negativeloglikelihood(x):
      return sum(
        (
          ((x[0] - x0[0][i]) / sigma[0][i]) ** 2
        + ((x[1] - x0[1][i]) / sigma[1][i]) ** 2
        + ((x[2] - x0[2][i]) / sigma[2][i]) ** 2
        ) for i in xrange(nbincontents)
      )
    return negativeloglikelihood

class OneParameterVVH(ConstrainedTemplatesWithFit):
  templatenames = "SM", "g13gi1", "g12gi2", "g11gi3", "BSM"
  pureindices = 0, 4

  @staticmethod
  def constraint(x):
    return np.array([minimizequartic(x)])

  constraintmin = np.finfo(np.float).eps
  constraintmax = np.inf

  def makeNLL(self, x0, sigma, nbincontents):
    def negativeloglikelihood(x):
      return sum(
        (
          ((x[0] - x0[0][i]) / sigma[0][i]) ** 2
        + ((x[1] - x0[1][i]) / sigma[1][i]) ** 2
        + ((x[2] - x0[2][i]) / sigma[2][i]) ** 2
        + ((x[3] - x0[3][i]) / sigma[3][i]) ** 2
        + ((x[4] - x0[4][i]) / sigma[4][i]) ** 2
        ) for i in xrange(nbincontents)
      )
    return negativeloglikelihood

  def bounds(self, fitstartpoint):
    maxevenstartpoint = max(_ for i, _ in enumerate(fitstartpoint) if i in (0, 2, 4))
    return optimize.Bounds(
      [np.finfo(float).eps, -10*maxevenstartpoint, -10*maxevenstartpoint, -10*maxevenstartpoint, np.finfo(float).eps],
      [2*fitstartpoint[0],   10*maxevenstartpoint,  10*maxevenstartpoint,  10*maxevenstartpoint, 2*fitstartpoint[4] ],
      keep_feasible=True,
    )
