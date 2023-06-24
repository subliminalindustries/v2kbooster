#! /usr/bin/env python

import sys
import argparse

import v2kbooster

parser = argparse.ArgumentParser(prog='v2kbooster',
                                 description='Boost v2k recordings to an intelligible level')

parser.add_argument('pattern', type=str, help='Glob pattern (ex.: "./*.wav")')

args = parser.parse_args(sys.argv[1:])

v2kbooster.process(args.pattern)