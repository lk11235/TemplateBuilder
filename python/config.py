import abc
import collections
import itertools
import json
import numbers
import os
import re

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
          "files": UniformJsonListOf(JsonStr),
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
          "weight": JsonStr,
        },
        defaultvalues = {
          "postprocessing": [],
          "selection": "1",
          "weight": "1",
        },
      ),
    ),
  }

  defaultvalues = {
    "constraints": [],
  }

class TemplateBuilder(object):
  def __init__(self, *filenames, **kwargs):
    self.__configs = [JsonReader(filename) for filename in filenames]
    self.__printbins = tuple(kwargs.pop("printbins", ()))
    self.__printallbins = kwargs.pop("printallbins", False)
    self.__force = kwargs.pop("force", False)
    self.__debug = kwargs.pop("debug", False)
    if kwargs:
      raise TypeError("Unknown kwargs: "+ ", ".join(kwargs))

  def maketemplates(self):
    templates = []
    constraints = []

    treeargs = set()
    for config in self.__configs:
      for templateconfig in config["templates"]:
        for filename in templateconfig["files"]:
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

    commonprefix = os.path.commonprefix(outfilenames)
    commonsuffix = os.path.commonprefix(list(_[::-1] for _ in outfilenames))[::-1]

    templatesbyname = {}

    with RootFiles(*outfilenames, commonargs=["RECREATE" if self.__force else "CREATE"]) as newfiles, opens(*logfilenames, commonargs="w") as logfiles:
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
              tree for tree in alltrees
                if tree.treename == templateconfig["tree"]
                and tree.filename in [
                  os.path.join(config["inputDirectory"], filename) for filename in templateconfig["files"]
                ]
            ]

            assert len(trees) == len(templateconfig["files"]), (len(trees), len(templateconfig["files"]))

            template = Template(
              templateconfig["name"],
              outfilename.replace(commonprefix, "", 1)[::-1].replace(commonsuffix[::-1], "", 1)[::-1],
              trees,
              templateconfig["variables"][0], templateconfig["binning"]["bins"][0], templateconfig["binning"]["bins"][1], templateconfig["binning"]["bins"][2],
              templateconfig["variables"][1], templateconfig["binning"]["bins"][3], templateconfig["binning"]["bins"][4], templateconfig["binning"]["bins"][5],
              templateconfig["variables"][2], templateconfig["binning"]["bins"][6], templateconfig["binning"]["bins"][7], templateconfig["binning"]["bins"][8],
              templateconfig["selection"], templateconfig["weight"],
              mirrortype, scaleby, floor
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

      for constraint in constraints:
        constraint.makefinaltemplates(printbins=self.__printbins, printallbins=self.__printallbins)

      for template in templates:
        if not template.finalized:
          raise RuntimeError("Never finalized {}".format(template.name))
