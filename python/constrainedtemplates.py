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

class OneParameterggH(ConstrainedTemplatesBase):
  ntemplates = 3  #pure SM, interference, pure BSM

  def __init__(self, *args, **kwargs):
    super(OneParameterggH, self).__init__(*args, **kwargs)
    if autograd is None:
      raise ImportError("To use OneParameterggH, please install autograd.")
    if not hasscipy:
      raise ImportError("To use OneParameterggH, please install a newer scipy.")

  def makefinaltemplates(self, printbins):
    SM, int, BSM = self.templates

    printbins = tuple(tuple(_) for _ in printbins)
    assert all(len(_) == 3 for _ in printbins)
    print
    print "Making the final templates:"
    print "  SM:  {:40} {:45}".format(SM.printprefix, SM.name)
    print "  int: {:40} {:45}".format(int.printprefix, int.name)
    print "  BSM: {:40} {:45}".format(BSM.printprefix, BSM.name)
    print "from individual templates with integrals:"

    for _ in SM, int, BSM:
      for component in _.templatecomponents:
        component.lock()
        print "  {:45} {:10.3e}".format(component.name, component.integral)

    printedbins = []

    for x, y, z in self.binsxyz:
      SMbincontent = {}
      for component in SM.templatecomponents:
        SMbincontent[component.name.replace(SM.name, "")] = component.GetBinContentError(x, y, z)

      intbincontent = {}
      for component in int.templatecomponents:
        intbincontent[component.name.replace(int.name, "")] = component.GetBinContentError(x, y, z)

      BSMbincontent = {}
      for component in BSM.templatecomponents:
        BSMbincontent[component.name.replace(BSM.name, "")] = component.GetBinContentError(x, y, z)

      #Each template component produces a 3D probability distribution in (SM, int, BSM)
      #FIXME: include correlations and don't approximate as Gaussian

      assert set(SMbincontent) == set(intbincontent) == set(BSMbincontent)
      assert len(SMbincontent) == len(SM.templatecomponents) == len(BSM.templatecomponents) == len(int.templatecomponents)

      x0SM, x0int, x0BSM, sigmaSM, sigmaint, sigmaBSM = [], [], [], [], [], []

      for name in SMbincontent:
        x0SM.append(SMbincontent[name].n)
        x0int.append(intbincontent[name].n)
        x0BSM.append(BSMbincontent[name].n)
        sigmaSM.append(SMbincontent[name].s)
        sigmaint.append(intbincontent[name].s)
        sigmaBSM.append(BSMbincontent[name].s)

      x0SM = np.array(x0SM)
      x0int = np.array(x0int)
      x0BSM = np.array(x0BSM)
      sigmaSM = np.array(sigmaSM)
      sigmaint = np.array(sigmaint)
      sigmaBSM = np.array(sigmaBSM)

      nbincontents = len(SMbincontent)

      def negativeloglikelihood(x):
        return sum(
          (
            ((x[0] - x0SM[i] ) / sigmaSM[i] ) ** 2
          + ((x[1] - x0int[i]) / sigmaint[i]) ** 2
          + ((x[2] - x0BSM[i]) / sigmaBSM[i]) ** 2
          ) for i in xrange(nbincontents)
        )

      nlljacobian = autograd.jacobian(negativeloglikelihood)
      nllhessian = autograd.hessian(negativeloglikelihood)

      #|interference| <= 2*sqrt(SM*BSM)
      #2*sqrt(SM*BSM) - |interference| >= 0

      def constraint(x):
        return np.array([2*(x[0]*x[2])**.5 - abs(x[1])])
      constraintjacobian = autograd.jacobian(constraint)
      constrainthessianv = autograd.linear_combination_of_hessians(constraint)

      nonlinearconstraint = optimize.NonlinearConstraint(constraint, np.finfo(np.float).eps, np.inf, constraintjacobian, constrainthessianv)

      linearconstraint = optimize.LinearConstraint([[1, 0, 0], [0, 0, 1]], [0, np.inf], [0, np.inf])

      startpoint = [weightedaverage(_.itervalues()).n for _ in SMbincontent, intbincontent, BSMbincontent]
      if startpoint[0] == 0: startpoint[0] = np.finfo(np.float).eps
      if startpoint[2] == 0: startpoint[2] = np.finfo(np.float).eps
      print [x, y, z], startpoint

      bounds = optimize.Bounds([np.finfo(float).eps, -np.inf, np.finfo(float).eps], [np.inf]*3)

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

      if fitresult.status != 0:
        warnings.warn(RuntimeWarning("Fit failed with status {}.  Message:\n{}".format(fitresult.status, fitresult.message)))

      SMfinalbincontent, intfinalbincontent, BSMfinalbincontent = fitresult.x

      if (x, y, z) in printbins:
        thingtoprint = "  {:3d} {:3d} {:3d}:".format(x, y, z)
        fmt = "      {:<%d} {:10.3e}" % max(len(name) for name in itertools.chain(SMbincontent, intbincontent, BSMbincontent))
        for name, content in itertools.chain(SMbincontent.iteritems(), intbincontent.iteritems(), BSMbincontent.iteritems()):
          thingtoprint += "\n"+fmt.format(name, content)
        thingtoprint += "\n"+fmt.format("final SM", finalSMbincontent)
        thingtoprint += "\n"+fmt.format("final int", finalintbincontent)
        thingtoprint += "\n"+fmt.format("final BSM", finalBSMbincontent)
        printedbins.append(thingtoprint)

      SM.SetBinContentError(x, y, z, SMfinalbincontent)
      int.SetBinContentError(x, y, z, intfinalbincontent)
      BSM.SetBinContentError(x, y, z, BSMfinalbincontent)

    if printedbins:
      print
      print "Bins you requested to print:"
      for _ in printedbins: print _

    SM.finalize()
    int.finalize()
    BSM.finalize()

    print
    print "final integrals:"
    print "  SM  = {:10.3e}".format(SM.integral)
    print "  BSM = {:10.3e}".format(BSM.integral)
    print "  int = {:10.3e}".format(int.integral)
    print
