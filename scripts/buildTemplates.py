#!/usr/bin/env python

from TemplateBuilder.TemplateBuilder.config import TemplateBuilder

if __name__ == "__main__":
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument("configfile", nargs="+")
  args = p.parse_args()

  TemplateBuilder(*args.configfile).maketemplates()
