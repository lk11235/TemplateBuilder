import abc
import collections
import json
import numbers
import os

import uncertainties

from rootfile import RootFile
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

  def maketemplates(self):
    templates = []

    treeargs = set()
    for templateconfig in self["templates"]:
      for filename in templateconfig["files"]:
        treeargs.add((os.path.join(self["inputDirectory"], filename), templateconfig["tree"]))
    alltrees = {Tree(*args) for args in treeargs}

    with RootFile(self["outputFile"], "CREATE") as newf:
      for templateconfig in self["templates"]:
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
              os.path.join(self["inputDirectory"], filename) for filename in templateconfig["files"]
            ]
        ]

        assert len(trees) == len(templateconfig["files"]), (len(trees), len(templateconfig["files"]))

        templates.append(
          Template(
            templateconfig["name"], trees,
            templateconfig["variables"][0], templateconfig["binning"]["bins"][0], templateconfig["binning"]["bins"][1], templateconfig["binning"]["bins"][2],
            templateconfig["variables"][1], templateconfig["binning"]["bins"][3], templateconfig["binning"]["bins"][4], templateconfig["binning"]["bins"][5],
            templateconfig["variables"][2], templateconfig["binning"]["bins"][6], templateconfig["binning"]["bins"][7], templateconfig["binning"]["bins"][8],
            templateconfig["selection"], templateconfig["weight"],
            mirrortype, scaleby, floor
          )
        )

      for tree in alltrees:
        with tree:
          tree.fillall()

      for template in templates:
        template.makefinaltemplate()
