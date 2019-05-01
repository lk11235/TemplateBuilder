import contextlib
import os

class RootFile(object):
  def __init__(self, filename, *args, **kwargs):
    self.__filename = filename
    self.__args = args
    self.__deleteifbad = kwargs.pop("deleteifbad", False)
    self.__entered = False
  def __enter__(self):
    import ROOT
    self.__bkpdirectory = ROOT.gDirectory.GetDirectory(ROOT.gDirectory.GetPath())
    self.__f = ROOT.TFile.Open(self.__filename, *self.__args)
    self.__entered = True
    if not self.__f:
      raise IOError(self.__filename+" is a null pointer, see above for details.")
    if self.IsZombie():
      self.__exit__()
      raise IOError(self.__filename+" is a zombie, see above for details.")

    try:
      openoption = self.__args[0].upper()
    except IndexError:
      openoption = ""

    self.__write = {
      "": False,
      "READ": False,
      "NEW": True,
      "CREATE": True,
      "RECREATE": True,
      "UPDATE": True,
    }[openoption]

    return self

  def __exit__(self, *errorstuff):
    if self.__write and (not any(errorstuff) or not self.__deleteifbad):
      self.Write()
    self.Close()
    self.__bkpdirectory.cd()
    if self.__write and self.__deleteifbad and any(errorstuff):
      os.remove(self.__filename)
  def __getattr__(self, attr):
    if self.__entered:
      return getattr(self.__f, attr)

@contextlib.contextmanager
def RootFiles(*filenames, **kwargs):
  if not filenames: yield []; return

  commonargs = kwargs.pop("commonargs", ())
  if kwargs: raise ValueError("Unknown kwargs: "+" ".join(kwargs))

  with RootFile(filenames[0], *commonargs) as f, RootFiles(*filenames[1:], commonargs=commonargs) as morefs:
    yield [f]+morefs

@contextlib.contextmanager
def opens(*filenames, **kwargs):
  if not filenames: yield []; return

  commonargs = kwargs.pop("commonargs", ())
  if kwargs: raise ValueError("Unknown kwargs: "+" ".join(kwargs))

  with open(filenames[0], *commonargs) as f, opens(*filenames[1:], commonargs=commonargs) as morefs:
    yield [f]+morefs

@contextlib.contextmanager
def RootCd(tdirectory, *args, **kwargs):
  import ROOT

  #https://root-forum.cern.ch/t/how-to-get-a-non-changing-copy-of-gdirectory-in-python/6236/2
  bkpdirectory = ROOT.gDirectory.GetDirectory(ROOT.gDirectory.GetPath())
  try:
    tdirectory.cd(*args, **kwargs)
    yield
  finally:
    bkpdirectory.cd()
