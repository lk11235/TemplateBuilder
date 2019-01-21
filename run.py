#!/usr/bin/env python

from config import JsonReader

if __name__ == "__main__":
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument("configfile")
  args = p.parse_args()

  JsonReader(args.configfile).maketemplates()
