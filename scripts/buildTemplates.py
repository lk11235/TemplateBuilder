#!/usr/bin/env python

from TemplateBuilder.TemplateBuilder.config import TemplateBuilder

if __name__ == "__main__":
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument("configfile", nargs="+")
  p.add_argument("--print-bin", action="append", default=[], type=int, nargs=3)
  p.add_argument("--print-all-bins", action="store_true")
  p.add_argument("--force", "-f", action="store_true")
  p.add_argument("--debug", action="store_true", help="only run 10000 events per tree.  files will be saved as (filename)_debug.root")
  args = p.parse_args()

  import warnings
  warnings.simplefilter("error")
  warnings.filterwarnings(action='ignore', category=RuntimeWarning, message=r'creating converter for unknown type "const char\*\[\]".*')

  TemplateBuilder(*args.configfile, printbins=args.print_bin, printallbins=args.print_all_bins, force=args.force, debug=args.debug).maketemplates()
