#!/usr/bin/env python

from TemplateBuilder.TemplateBuilder.config import TemplateBuilder

if __name__ == "__main__":
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument("configfile", nargs="+")
  p.add_argument("--print-bin", action="append", default=[], type=int, nargs=3)
  p.add_argument("--force", "-f", action="store_true")
  args = p.parse_args()

  TemplateBuilder(*args.configfile, printbins=args.print_bin, force=args.force).maketemplates()
