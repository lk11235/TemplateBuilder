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

from moreuncertainties import weightedaverage


def ConstrainedTemplates(constrainttype, templates):
  return {
    "unconstrained": OneTemplate,
    "oneparameterggH": OneParameterggH,
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

  @abc.abstractproperty
  def ntemplates(self): pass

  @abc.abstractmethod
  def makefinaltemplates(self, printbins): pass

  @property
  def binsxyz(self):
    for binxyz in itertools.izip(*(t.binsxyz for t in self.templates)):
      binxyz = set(binxyz)
      if len(binxyz) != 1:
        raise ValueError("Templates have inconsistent binning")
      yield binxyz.pop()

class OneTemplate(ConstrainedTemplatesBase):
  ntemplates = 1

  def makefinaltemplates(self, printbins):
    template, = self.templates

    printbins = tuple(tuple(_) for _ in printbins)
    assert all(len(_) == 3 for _ in printbins)
    print
    print "Making the final template:"
    print "  {:40} {:45}".format(template.printprefix, template.name)
    print "from individual templates with integrals:"

    for component in template.templatecomponents:
      component.lock()
      print "  {:45} {:10.3e}".format(component.name, component.integral)

    outlierwarning = []
    printedbins = []

    for x, y, z in self.binsxyz:
      bincontent = {}
      for component in template.templatecomponents:
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

      if len(bincontent) < len(template.templatecomponents) / 2.:
        raise RuntimeError("Removed more than half of the bincontents!  Please check.\n" + "\n".join("  {:45} {:10.3e}".format(component.name, component.GetBinContentError(x, y, z)) for component in template.templatecomponents))

      if all(_.n == _.s == 0 for _ in bincontent.itervalues()):  #special case, empty histogram
        finalbincontent = bincontent.values()[0]
      else:                                                      #normal case
        finalbincontent = weightedaverage(bincontent.itervalues())

      if (x, y, z) in printbins:
        thingtoprint = "  {:3d} {:3d} {:3d}:".format(x, y, z)
        fmt = "      {:<%d} {:10.3e}" % max(len(name) for name in bincontent)
        for name, content in bincontent.iteritems():
          thingtoprint += "\n"+fmt.format(name, content)
        thingtoprint += "\n"+fmt.format("final", finalbincontent)
        printedbins.append(thingtoprint)

      template.SetBinContentError(x, y, z, finalbincontent)

    if outlierwarning:
      print
      print "Warning: there are outliers in some bins:"
      for _ in outlierwarning: print _

    if printedbins:
      print
      print "Bins you requested to print:"
      for _ in printedbins: print _

    template.finalize()

    print
    print "final integral = {:10.3e}".format(template.integral)
    print

class ConstraintedTemplatesWithFit(ConstrainedTemplatesBase):
  def __init__(self, *args, **kwargs):
    super(ConstraintedTemplatesWithFit, self).__init__(*args, **kwargs)
    if autograd is None:
      raise ImportError("To use OneParameterggH, please install autograd.")
    if not hasscipy:
      raise ImportError("To use OneParameterggH, please install a newer scipy.")

  @abc.abstractproperty
  def templatenames(self): "can be a class member, names of the templates in order (e.g. SM, int, BSM)"

  @abc.abstractmethod
  def makeNLL(self, x0, sigma, nbincontents): pass

  @abc.abstractmethod
  def constraint(self, x): "can be static"

  @abc.abstractmethod
  def pureindices(self): "can be a class member"

  def makefinaltemplates(self, printbins):
    printbins = tuple(tuple(_) for _ in printbins)
    assert all(len(_) == 3 for _ in printbins)
    print
    print "Making the final templates:"
    for name, _ in itertools.izip(self.templatenames, self.templates):
      print "  {:>10}: {:40} {:45}".format(name, _.printprefix, _.name)
    print "from individual templates with integrals:"

    for _ in self.templates:
      for component in _.templatecomponents:
        component.lock()
        print "  {:45} {:10.3e}".format(component.name, component.integral)

    printedbins = []

    for x, y, z in self.binsxyz:
      bincontent = []
      for _ in self.templates:
        thisonescontent = {}
        bincontent.append(thisonescontent)
        for component in _.templatecomponents:
          thisonescontent[component.name.replace(_.name, "")] = component.GetBinContentError(x, y, z)

      nbincontents = len(bincontent[0])

      #Each template component produces a 3D probability distribution in (SM, int, BSM)
      #FIXME: include correlations and don't approximate as Gaussian

      assert len({frozenset(_) for _ in bincontent}) == 1  #they should all have the same keys

      x0 = [[] for t in self.templates]
      sigma = [[] for t in self.templates]

      for name in bincontent[0]:
        for thisonescontent, thisx0, thissigma in itertools.izip(bincontent, x0, sigma):
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

      nonlinearconstraint = optimize.NonlinearConstraint(constraint, np.finfo(np.float).eps, np.inf, constraintjacobian, constrainthessianv)

      bounds = optimize.Bounds(
        [np.finfo(float).eps if i in self.pureindices else -np.inf for i in xrange(self.ntemplates)],
        [np.inf for i in xrange(self.ntemplates)]
      )

      startpoint = [weightedaverage(_.itervalues()).n for _ in bincontent]
      for i in self.pureindices:
        if startpoint[i] == 0: startpoint[i] = np.finfo(np.float).eps
      print [x, y, z], startpoint

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

      finalbincontent = fitresult.x

      if fitresult.status != 0:
        warnings.warn(RuntimeWarning("Fit gave status {}.  Message:\n{}".format(fitresult.status, fitresult.message)))

      if (x, y, z) in printbins:
        thingtoprint = "  {:3d} {:3d} {:3d}:".format(x, y, z)
        fmt = "      {:<%d} {:10.3e}" % max(len(name) for name in itertools.chain(*bincontent))
        for name, content in itertools.chain(*(_.iteritems() for _ in bincontent)):
          thingtoprint += "\n"+fmt.format(name, content)
        for name, content in itertools.izip(self.templatenames, finalbinconent):
          thingtoprint += "\n"+fmt.format("final "+name, finalSMbincontent)
        printedbins.append(thingtoprint)

      for t, content in itertools.izip(self.templates, finalbincontent):
        t.SetBinContentError(x, y, z, content)

    if printedbins:
      print
      print "Bins you requested to print:"
      for _ in printedbins: print _

    for _ in self.templates:
      _.finalize()

    print
    print "final integrals:"
    for name, t in itertools.izip(self.templatenames, self.templates):
      print "  {:>10} = {:10.3e}".format(name, t.integral)
    print


class OneParameterggH(ConstraintedTemplatesWithFit):
  ntemplates = 3  #pure SM, interference, pure BSM
  templatenames = "SM", "int", "BSM"
  pureindices = 0, 2

  @staticmethod
  def constraint(x):
    #|interference| <= 2*sqrt(SM*BSM)
    #2*sqrt(SM*BSM) - |interference| >= 0
    return np.array([2*(x[0]*x[2])**.5 - abs(x[1])])

  def makeNLL(self, x0, sigma, nbincontents):
    def negativeloglikelihood(x):
      return sum(
        (
          ((x[0] - x0[0][i] ) / sigma[0][i] ) ** 2
        + ((x[1] - x0[1][i]) / sigma[1][i]) ** 2
        + ((x[2] - x0[2][i]) / sigma[2][i]) ** 2
        ) for i in xrange(nbincontents)
      )
    return negativeloglikelihood

