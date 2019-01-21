import os

class RootFile(object):
  def __init__(self, filename, *args):
    self.__filename = filename
    self.__args = args
    self.__entered = False
  def __enter__(self):
    import ROOT
    self.__f = ROOT.TFile.Open(self.__filename, *self.__args)
    self.__entered = True
    if not self.__f:
      raise IOError(self.__filename+" is a null pointer, see above for details.")
    if self.IsZombie():
      self.__exit__()
      raise IOError(self.__filename+" is a zombie, see above for details.")
    self.__write = bool(self.GetBytesWritten())
    return self
  def __exit__(self, *errorstuff):
    if self.__write and not any(errorstuff):
      self.Write()
    self.Close()
    if self.__write and any(errorstuff):
      os.remove(self.__filename)
  def __getattr__(self, attr):
    if self.__entered:
      return getattr(self.__f, attr)
