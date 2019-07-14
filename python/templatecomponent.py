import abc, itertools

class TemplateComponentBase(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def templatecomponentpieces(self): pass

class TemplateComponent(TemplateComponentBase):
  def __init__(
    self, name, trees, subdirectories,
    printprefix,
    xformula, xbins, xmin, xmax,
    yformula, ybins, ymin, ymax,
    zformula, zbins, zmin, zmax,
    cutformula, weightformula,
    mirrortype, scaleby
  ):
    self.__templatecomponentpiecehandles = [
      tree.registertemplatecomponentpiece(
        name+"_"+subdirectory, printprefix,
        xformula, xbins, xmin, xmax,
        yformula, ybins, ymin, ymax,
        zformula, zbins, zmin, zmax,
        cutformula, weightformula,
        mirrortype, scaleby,
        subdirectory=subdirectory,
      )
      for i, (tree, subdirectory) in enumerate(itertools.izip(trees, subdirectories))
    ]

  @property
  def templatecomponentpieces(self):
    return [handle() for handle in self.__templatecomponentpiecehandles]

  @property
  def rootless(self):
    return RootlessTemplateComponent(self.templatecomponentpieces)

class RootlessTemplateComponent(TemplateComponentBase):
  def __init__(self, templatecomponentpieces):
    self.__templatecomponentpieces = [_.rootless for _ in templatecomponentpieces]
  @property
  def templatecomponentpieces(self):
    return self.__templatecomponentpieces
