import array
import collections
import functools
import itertools
import re

import numpy as np
import uncertainties

import ROOT

TreeVariable = collections.namedtuple("TreeVariable", "formula nbins min max")

class TemplateComponent(object):
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
