import multiprocessing

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

