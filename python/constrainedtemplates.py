import abc, itertools, warnings

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

  def makefinaltemplates(self, printbins):
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
      bincontents = []
      for _ in self.templates:
        thisonescontent = {}
        bincontents.append(thisonescontent)
        for component in _.templatecomponents:
          thisonescontent[component.name.replace(_.name, "")] = component.GetBinContentError(x, y, z)

      assert len({frozenset(_) for _ in bincontents}) == 1  #they should all have the same keys

      try:
        finalbincontents, printmessage, warning = self.computefinalbincontents(bincontents)
      except:
        print "Error when finding content for bin", x, y, z
        raise

      printmessage = "  {:3d} {:3d} {:3d}:\n".format(x, y, z) + printmessage
      if (x, y, z) in printbins:
        printedbins.append(printmessage)

      if warning:
        warnings.append("  {:3d} {:3d} {:3d}: ".format(x, y, z) + warning)

      for t, content in itertools.izip(self.templates, finalbincontents):
        t.SetBinContentError(x, y, z, content)

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

class OneTemplate(ConstrainedTemplatesBase):
  templatenames = "",
  pureindices = 0, 2

  def computefinalbincontents(self, bincontents):
    namestoremove = set()
    bincontent = bincontents[0]
    nbincontents = len(bincontent)

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

    outlierwarning = None
    if namestoremove:
      outlierwarning = "there are outliers: " + ", ".join(sorted(namestoremove))
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

    return [finalbincontent], thingtoprint, outlierwarning

class ConstrainedTemplatesWithFit(ConstrainedTemplatesBase):

  def computefinalbincontents(self, bincontents):
    bincontents = bincontents[:]
    nbincontents = len(bincontents[0])

    #Each template component produces a 3D probability distribution in (SM, int, BSM)
    #FIXME: include correlations and don't approximate as Gaussian

    x0 = [[] for t in self.templates]
    sigma = [[] for t in self.templates]

    for name in bincontents[0]:
      for thisonescontent, thisx0, thissigma in itertools.izip(bincontents, x0, sigma):
        if thisonescontent[name].n == 0:
          thisonescontent[name].std_dev = max(thisonescontent[othername].s for othername in thisonescontent)
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

    bounds = optimize.Bounds(
      [np.finfo(float).eps if i in self.pureindices else -np.inf for i in xrange(self.ntemplates)],
      [np.inf for i in xrange(self.ntemplates)]
    )

    startpoint = [weightedaverage(_.itervalues()).n for _ in bincontents]
    for i in self.pureindices:
      if startpoint[i] == 0: startpoint[i] = np.finfo(np.float).eps

    fitresult = optimize.minimize(
      negativeloglikelihood,
      startpoint,
      method='trust-constr',
      jac=nlljacobian,
      hess=nllhessian,
      constraints=[nonlinearconstraint],
      bounds=bounds,
      options = {},
    )

    print fitresult

    finalbincontents = fitresult.x

    warning = None
    if fitresult.status != 0:
      warning = "Fit gave status {}.  Message:\n{}".format(fitresult.status, fitresult.message)

    thingtoprint = ""
    fmt = "      {:<%d} {:10.3e}" % max(len(name) for name in itertools.chain(*bincontents))
    for name, content in itertools.chain(*(_.iteritems() for _ in bincontents)):
      thingtoprint += "\n"+fmt.format(name, content)
    for name, content in itertools.izip(self.templatenames, finalbincontents):
      thingtoprint += "\n"+fmt.format("final "+name, content)

    return finalbincontents, thingtoprint, warning

  def __init__(self, *args, **kwargs):
    super(ConstrainedTemplatesWithFit, self).__init__(*args, **kwargs)
    if autograd is None:
      raise ImportError("To use OneParameterggH, please install autograd.")
    if not hasscipy:
      raise ImportError("To use OneParameterggH, please install a newer scipy.")

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

