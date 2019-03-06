import abc, itertools

from moreuncertainties import weightedaverage

def ConstrainedTemplates(constrainttype, templates):
  return {
    "unconstrained": OneTemplate,
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

class OneTemplate(ConstrainedTemplatesBase):
  ntemplates = 1

  def makefinaltemplates(self, printbins):
    assert len(self.templates) == 1
    template = self.templates[0]

    printbins = tuple(tuple(_) for _ in printbins)
    assert all(len(_) == 3 for _ in printbins)
    print
    print "Making the final template:"
    print "  {:40} {:45}".format(template.printprefix, template.name)
    print "from individual templates with integrals:"

    for component in template.templatecomponents:
      component.lock()
      print "  {:45} {:10.3e}".format(component.name, component.integral)

    flooredbins = []
    outlierwarning = []
    printedbins = []

    for x, y, z in template.binsxyz:
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

    template.doscale()
    template.domirror()
    template.dofloor()

    print
    print "final integral = {:10.3e}".format(template.integral)
    print

    
