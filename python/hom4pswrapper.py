import abc, contextlib, multiprocessing, os, re, subprocess

import numpy as np

nproc = multiprocessing.cpu_count()

def getcommandline(dispatch=None, engine=None, threads=None, pre=["balance", "const"], post=["pop", "pop", "refine", "summary"], mprec=None, summarydigits=None, homotopy="polyexp"):
  args = ["hom4ps-core"]
  if dispatch is not None: args.append("--dispatch="+dispatch)
  if engine is not None: args.append("--engine="+engine)
  if mprec is not None:
    args.append("-m{:d}".format(mprec))
  if threads is not None:
    if threads.upper() == "CPU": threads = str(nproc)
    args.append("--threads="+threads)
  for _ in pre:
    args.append("--pre="+_)
  args += [
    "--sink=mem",
  ]
  if "homotopy" is not None:
    args.append("--homotopy="+homotopy)
  args += [
    "--monitor=facet-div"
  ]
  for _ in post:
    args.append("--post="+_)
  args += [
    "-fsummary-standard:1",
    "-fsummary-digits:{:d}".format(summarydigits),
  ]
  return args


def smallcmdline():
  return getcommandline(dispatch="serial", engine="stack", summarydigits=16)
def fastcmdline():
  return getcommandline(threads="CPU", summarydigits=16)
def smallparallelcmdline():
  return getcommandline(dispatch="serial", engine="stack", summarydigits=16, threads="CPU")
def smallparalleltdegcmdline():
  return getcommandline(dispatch="serial", engine="stack", summarydigits=16, threads="CPU", homotopy="tdeg")
def smallparalleltdegnopostcmdline():
  return getcommandline(dispatch="serial", engine="stack", summarydigits=16, threads="CPU", homotopy="tdeg", post=["summary"])
def easycmdline():
  return getcommandline(mprec=140, threads="CPU", pre=["bouncer", "balance", "const"], post=["pop", "pop", "refine-mp", "summary"], summarydigits=32)

def getcmdline(which):
  return {
    "small": smallcmdline,
    "fast": fastcmdline,
    "smallparallel": smallparallelcmdline,
    "smallparalleltdeg": smallparallelcmdline,
    "smallparalleltdegnopost": smallparalleltdegnopostcmdline,
    "easy": easycmdline,
  }[which]()

class Hom4PSResult(object):
  superinitargs = ()
  def __init__(self, stdin, stdout, stderr):
    self.stdin = stdin
    self.stdout = stdout
    self.stderr = stderr
    super(Hom4PSResult, self).__init__(*self.superinitargs)
  @property
  def nfailedpaths(self):
    match = re.search(r"Failed Paths\s*:\s*([0-9]+)", self.stdout)
    try:
      return int(match.group(1))
    except AttributeError:
      print "-----------------------"
      print self.stdout
      print "-----------------------"
      raise
  @property
  def ndivergentpaths(self):
    match = re.search(r"Divergent:\s*([0-9]+)", self.stdout)
    try:
      return int(match.group(1))
    except AttributeError:
      print "-----------------------"
      print self.stdout
      print "-----------------------"
      raise
  @property
  def solutions(self):
    result = []
    for solution in self.stdout.split("\n\n"):
      if not re.search("solution *#", solution): continue
      match = re.search(r"values: *([^\n]*)", solution)
      if "This solution appears to be real" in solution:
        result.append(np.array([float(_) for _ in match.group(1).split()]))
      else:
        result.append(np.array([complex(re.sub(r"i[*]([0-9Ee.+-]+)", r"\1j", _).replace("+-", "-")) for _ in match.group(1).split()]))
    return result
  @property
  def realsolutions(self):
    return [solution for solution in self.solutions if np.all(np.isreal(solution))]

class Hom4PSRuntimeError(Hom4PSResult, RuntimeError):
  @abc.abstractproperty
  def errormessage(self): pass
  @property
  def superinitargs(self): return self.errormessage+"\n\ninput:\n\n"+self.stdin+"\n\nstdout:\n\n"+self.stdout+"\n\nstderr:\n\n"+self.stderr,

class Hom4PSErrorMessage(Hom4PSRuntimeError):
  errormessage = "hom4ps printed an error message."

class Hom4PSFailedPathsError(Hom4PSRuntimeError):
  errormessage = "hom4ps found some failed paths."
  def __init__(self, *args, **kwargs):
    super(Hom4PSFailedPathsError, self).__init__(*args, **kwargs)
    assert self.nfailedpaths != 0

class Hom4PSDivergentPathsError(Hom4PSRuntimeError):
  errormessage = "hom4ps found some divergent paths."
  def __init__(self, *args, **kwargs):
    super(Hom4PSDivergentPathsError, self).__init__(*args, **kwargs)
    assert self.ndivergentpaths != 0

class Hom4PSDivergentAndFailedPathsError(Hom4PSFailedPathsError, Hom4PSDivergentPathsError):
  errormessage = "hom4ps found some divergent paths and some failed paths."

@contextlib.contextmanager
def setenv(name, value):
  oldvalue = os.environ.get(name, None)
  os.environ[name] = value
  try:
    yield
  finally:
    if oldvalue is None:
      del os.environ[name]
    else:
      os.environ[name] = oldvalue

@contextlib.contextmanager
def addtocpath(path):
  oldvalue = os.environ.get("CPATH", None)
  if oldvalue is None:
    value = path
  else:
    value = oldvalue+":"+path
  with setenv("CPATH", value):
    yield

def runhom4ps(stdin, whichcmdline, verbose=False):
  cmdline = getcmdline(whichcmdline)
  if verbose: print stdin
  with addtocpath(os.path.join(os.environ["CMSSW_BASE"], "include")):
    p = subprocess.Popen(cmdline, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate(stdin)
  if "error" in err:
    raise Hom4PSErrorMessage(stdin, out, err)
  result = Hom4PSResult(stdin, out, err)
  if result.nfailedpaths or result.ndivergentpaths:
    if verbose: print out
    if not result.nfailedpaths:
      raise Hom4PSDivergentPathsError(stdin, out, err)
    elif not result.ndivergentpaths:
      raise Hom4PSFailedPathsError(stdin, out, err)
    else:
      raise Hom4PSDivergentAndFailedPathsError(stdin, out, err)
  if verbose: print out
  return result
