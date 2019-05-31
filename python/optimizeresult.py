import scipy.optimize

def indent(text, prefix, predicate=None):
  return ''.join(
    prefix+line if predicate is None or predicate(line) else line
    for line in text.splitlines(True)
  )

class OptimizeResult(scipy.optimize.OptimizeResult):
  def __formatvforrepr(self, v, m):
    if isinstance(v, OptimizeResult):
      return "\n" + indent(repr(v), ' '*m)
    return repr(v)

  def __repr__(self):
    if self.keys():
      m = max(map(len, list(self.keys()))) + 1
      return '\n'.join([k.rjust(m) + ': ' + self.__formatvforrepr(v, m)
                        for k, v in sorted(self.items())])
    else:
      return self.__class__.__name__ + "()"

