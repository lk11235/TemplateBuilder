import abc
import collections
import itertools
import json
import numbers
import os
import re

import numpy as np
import uncertainties

from constrainedtemplates import ConstrainedTemplates
from fileio import opens, RootCd, RootFiles
from tree import Tree
from template import Template

class JsonBase(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, json, nameforerror):
    self.__json = json
    self.__nameforerror = nameforerror

    if not isinstance(json, self.maintype):
      raise TypeError(self.nameforerror+" is supposed to be {.__name__}, but instead it's {.__name__}".format(self.maintype, type(json)))

  @property
  def nameforerror(self): return self.__nameforerror
  @property
  def json(self): return self.__json

  @abc.abstractmethod
  def maintype(self): "override with a class member variable"

  def __eq__(self, other):
    return self.json == other
  def __ne__(self, other):
    return not self == other

def JsonStr(object, nameforerror):
  if not isinstance(object, basestring):
    raise ValueError("{} is not a string".format(nameforerror))
  return object

def JsonInt(object, nameforerror):
  if not isinstance(object, numbers.Integral):
    raise ValueError("{} is not an integer".format(nameforerror))
  return object

def JsonBool(object, nameforerror):
  if not isinstance(object, bool):
    raise ValueError("{} is not a bool".format(nameforerror))
  return object

def JsonFloat(object, nameforerror):
  if not isinstance(object, numbers.Real):
    raise ValueError("{} is not a float".format(nameforerror))
  return object

class JsonList(JsonBase):
  maintype = list
  def getitem(self, item, typ):
    try:
      return typ(self.json[item], nameforerror="{}[{}]".format(self.nameforerror, item))
    except KeyError:
      raise KeyError("No {} in {.nameforerror}".format(item, self))
    except TypeError as e:
      raise TypeError("When getting {} (type {.__name__}) from {.nameforerror}:\n\n{}".format(item, typ, self, e))

  def __len__(self):
    return len(self.json)
  def __contains__(self, other):
    return any(other == _ for _ in self)

  def append(self, item):
    self.json.append(item)
    type(self)(self.json, self.nameforerror)

class JsonListWithFormat(JsonList, collections.Sequence):
  def __init__(self, *args, **kwargs):
    result = super(JsonListWithFormat, self).__init__(*args, **kwargs)
    if len(self) != len(self.format):
      raise IndexError("{.nameforerror} is the wrong length ({} instead of {})".format(self, len(self), len(self.format)))
  @abc.abstractmethod
  def format(self): "override with a class member variable.  example: [int, float, float]"
  def getitem(self, item):
    return super(JsonListWithFormat, self).getitem(item, self.format[item])
  def __getitem__(self, item):
    return self.getitem(item)

def JsonListWithThisFormat(clsname, format):
  theformat = format
  class JsonListWithThisFormat(JsonListWithFormat):
    format = theformat
  JsonListWithThisFormat.__name__ = clsname
  return JsonListWithThisFormat

class UniformJsonList(JsonList, collections.Sequence):
  @abc.abstractmethod
  def subtype(self): "override with a class member variable"
  def getitem(self, item):
    return super(UniformJsonList, self).getitem(item, self.subtype)
  def __getitem__(self, item):
    return self.getitem(item)

def UniformJsonListOf(typ):
  class UniformJsonListOf(UniformJsonList):
    subtype = typ
    if not isinstance(subtype, type): subtype = staticmethod(subtype)
  UniformJsonListOf.__name__ += typ.__name__
  return UniformJsonListOf

class JsonDict(JsonBase):
  maintype = dict
  def getitem(self, key, typ, default=NotImplemented):
    try:
      return typ(
        self.json[key] if default is NotImplemented else self.json.get(key, default),
        nameforerror="{}[{}]".format(self.nameforerror, key)
      )
    except KeyError:
      raise KeyError("No {} in {.nameforerror}".format(key, self))
    except TypeError as e:
      raise TypeError("When getting {} (type {.__name__}) from {.nameforerror}:\n\n{}".format(key, typ, self, e))

class JsonDictWithFormat(collections.Mapping, JsonDict):
  def __init__(self, *args, **kwargs):
    result = super(JsonDictWithFormat, self).__init__(*args, **kwargs)
    for key in self.json:
      if key not in self.format:
        raise ValueError("Unknown key {} in {.nameforerror}".format(key, self))
  @abc.abstractmethod
  def format(self): "override with a class member variable.  e.g. {'intkey': int, 'listkey': JsonList}"
  defaultvalues = {}
  def getitem(self, key):
    return super(JsonDictWithFormat, self).getitem(key, self.format[key], default=self.defaultvalues.get(key, NotImplemented))
  def __getitem__(self, item):
    return self.getitem(item)
  def __len__(self):
    return len(self.format)
  def __iter__(self):
    return iter(self.format)

def JsonDictWithThisFormat(clsname, format, defaultvalues=None):
  theformat = format
  thedefaultvalues = defaultvalues
  if thedefaultvalues is None: thedefaultvalues = {}
  class JsonDictWithThisFormat(JsonDictWithFormat):
    format = theformat
    defaultvalues = thedefaultvalues
  JsonDictWithThisFormat.__name__ = clsname
  return JsonDictWithThisFormat

class JsonReader(JsonDictWithFormat):
  def __init__(self, filename):
    self.__filename = filename
    with open(filename) as f:
      super(JsonReader, self).__init__(json.load(f), filename)

  @property
  def filename(self):
    return self.__filename

  format = {
    "inputDirectory": JsonStr,
    "outputFile": JsonStr,
    "constraints": UniformJsonListOf(
      JsonDictWithThisFormat(
        "JsonConstraintConfig",
        format={
          "type": JsonStr,
          "templates": UniformJsonListOf(JsonStr),
        },
      ),
    ),
    "templates": UniformJsonListOf(
      JsonDictWithThisFormat(
        "JsonTemplateConfig",
        format={
          "binning": JsonDictWithThisFormat(
            "JsonTemplateBinningConfig",
            format={
              "bins": JsonListWithThisFormat(
                 "JsonTemplateBinningBinsConfig",
                 format=[JsonInt, JsonFloat, JsonFloat, JsonInt, JsonFloat, JsonFloat, JsonInt, JsonFloat, JsonFloat]
              ),
            },
          ),
          "files": UniformJsonListOf(
            UniformJsonListOf(JsonStr),
          ),
          "name": JsonStr,
          "postprocessing": UniformJsonListOf(
            JsonDictWithThisFormat(
              "JsonTemplatePostprocessingConfig",
              format={
                "type": JsonStr,
                "factor": JsonFloat,
                "factorerror": JsonFloat,
                "antisymmetric": JsonBool,
                "floorvalue": JsonFloat,
                "floorerror": JsonFloat,
              },
              defaultvalues={
              },
            ),
          ),
          "selection": JsonStr,
          "tree": JsonStr,
          "variables": JsonListWithThisFormat(
            "JsonTemplateVariablesConfig",
            format=[JsonStr, JsonStr, JsonStr],
          ),
          "weight": UniformJsonListOf(JsonStr),
        },
        defaultvalues = {
          "postprocessing": [],
          "selection": "1",
          "weight": ["1"],
        },
      ),
    ),
  }

  defaultvalues = {
    "constraints": [],
    "inputDirectory": os.getcwd(),
  }

class TemplateBuilder(object):
  def __init__(self, *filenames, **kwargs):
    self.__configs = [JsonReader(filename) for filename in filenames]
    self.__printbins = tuple(kwargs.pop("printbins", ()))
    self.__printallbins = kwargs.pop("printallbins", False)
    self.__force = kwargs.pop("force", False)
    self.__debug = kwargs.pop("debug", False)
    self.__useexistingcomponents = kwargs.pop("useexistingcomponents", False)
    self.__useexistingtemplates = kwargs.pop("useexistingtemplates", False)
    self.__binsortkey = kwargs.pop("binsortkey", None)
    self.__nthreads = kwargs.pop("nthreads", 1)
    if kwargs:
      raise TypeError("Unknown kwargs: "+ ", ".join(kwargs))
    if self.__useexistingtemplates and not self.__useexistingcomponents:
      raise ValueError("Can't do useexistingtemplates without useexistingcomponents")

  def maketemplates(self):
    templates = []
    constraints = []

    treeargs = set()
    for config in self.__configs:
      for templateconfig in config["templates"]:
        for listoffilenames in templateconfig["files"]:
          for filename in listoffilenames:
            treeargs.add((os.path.join(config["inputDirectory"], filename), templateconfig["tree"]))
    alltrees = {Tree(*args, debug=self.__debug) for args in treeargs}

    outfilenames = [config["outputFile"] for config in self.__configs]
    for outfilename in outfilenames:
      if not outfilename.endswith(".root"):
        raise ValueError(outfilename+" doesn't end with .root")
      if self.__debug:
        outfilenames = [re.sub("[.]root$", "_debug.root", outfilename) for outfilename in outfilenames]
    logfilenames = [re.sub("[.]root$", ".log", outfilename) for outfilename in outfilenames]
    assert not set(logfilenames) & set(outfilenames), set(logfilenames) & set(outfilenames)
    donefilenames = [re.sub("[.]root$", ".done", outfilename) for outfilename in outfilenames]
    assert not set(donefilenames) & set(outfilenames), set(donefilenames) & set(outfilenames)

    for outfilename, logfilename, donefilename in itertools.izip(outfilenames[:], logfilenames[:], donefilenames[:]):
      if os.path.exists(logfilename) and not os.path.exists(outfilename):
        os.remove(logfilename)
      if os.path.exists(donefilename) and (not os.path.exists(outfilename) or not self.__useexistingtemplates):
        os.remove(donefilename)

      if os.path.exists(donefilename):
        outfilenames.remove(outfilename)
        logfilenames.remove(logfilename)
        donefilenames.remove(donefilename)

    commonprefix = os.path.commonprefix(outfilenames)
    commonsuffix = os.path.commonprefix(list(_[::-1] for _ in outfilenames))[::-1]

    templatesbyname = {}

    fileopenoption = "CREATE"
    if self.__force:
      fileopenoption = "RECREATE"
    if self.__useexistingcomponents:
      fileopenoption = "UPDATE"

    with RootFiles(*outfilenames, commonargs=[fileopenoption]) as newfiles, opens(*logfilenames, commonargs="a") as logfiles:
      for config, outfilename, outfile, logfile in itertools.izip(self.__configs, outfilenames, newfiles, logfiles):
        with RootCd(outfile):
          for templateconfig in config["templates"]:
            mirrortype = None
            scaleby = None
            floor = None
            for postprocessing in templateconfig["postprocessing"]:
              if postprocessing["type"] == "mirror":
                if mirrortype is not None: raise ValueError("Multiple mirror blocks in {.nameforerror}".format(postprocessing))
                mirrortype = "antisymmetric" if postprocessing.get("antisymmetric", False) else "symmetric"
              elif postprocessing["type"] == "rescale":
                if scaleby is not None: raise ValueError("Multiple rescale blocks in {.nameforerror}".format(postprocessing))
                scaleby = uncertainties.ufloat(postprocessing["factor"], postprocessing.get("factorerror", 0))
              elif postprocessing["type"] == "floor":
                if floor is not None: raise ValueError("Multiple floor blocks in {.nameforerror}".format(postprocessing))
                floor = uncertainties.ufloat(postprocessing["floorvalue"], postprocessing.get("floorerror", 0))

            trees = [
              [
                tree for tree in alltrees
                  if tree.treename == templateconfig["tree"]
                  and tree.filename in [
                    os.path.join(config["inputDirectory"], filename) for filename in listoffilenames
                  ]
              ] for listoffilenames in templateconfig["files"]
            ]

            if len(templateconfig["weight"]) != len(trees):
              raise ValueError("You've provided {} lists of filenames, but {} weights".format(len(trees), len(templateconfig["weight"])))

            template = Template(
              templateconfig["name"],
              outfilename.replace(commonprefix, "", 1)[::-1].replace(commonsuffix[::-1], "", 1)[::-1],
              trees,
              templateconfig["variables"][0], templateconfig["binning"]["bins"][0], templateconfig["binning"]["bins"][1], templateconfig["binning"]["bins"][2],
              templateconfig["variables"][1], templateconfig["binning"]["bins"][3], templateconfig["binning"]["bins"][4], templateconfig["binning"]["bins"][5],
              templateconfig["variables"][2], templateconfig["binning"]["bins"][6], templateconfig["binning"]["bins"][7], templateconfig["binning"]["bins"][8],
              templateconfig["selection"], templateconfig["weight"],
              mirrortype, scaleby, floor,
              reuseifexists=self.__useexistingtemplates,
            )

            templates.append(template)

            if not any(template.name in constraintconfig["templates"] for constraintconfig in config["constraints"]):
              config["constraints"].append({"type": "unconstrained", "templates": [template.name]})

            templatesbyname[template.name] = template

          for constraintconfig in config["constraints"]:
            constrainttype = constraintconfig["type"]
            constrainedtemplates = []
            for name in constraintconfig["templates"]:
              try:
                constrainedtemplates.append(templatesbyname.pop(name))
              except KeyError:
                raise ValueError("Trying to use "+name+" for a constraint, but didn't find this template.  (Or maybe it's used for multiple constraints.  Don't do that.)")
            constraints.append(ConstrainedTemplates(constraintconfig["type"], constrainedtemplates, logfile=logfile))

      for tree in alltrees:
        with tree:
          tree.fillall()

      for template in templates:
        for component in template.templatecomponents:
          for piece in component.templatecomponentpieces:
            piece.lock()

      for constraint in sorted(constraints, key=lambda x: x.ntemplates): #do the ones with fewer templates first because they're less likely to fail, so we won't have to remake their components if the later ones fail
        constraint.makefinaltemplates(printbins=self.__printbins, printallbins=self.__printallbins, binsortkey=self.__binsortkey, nthreads=self.__nthreads)

      for template in templates:
        if not template.finalized:
          raise RuntimeError("Never finalized {}".format(template.name))

    with opens(*donefilenames, commonargs="w"): pass
