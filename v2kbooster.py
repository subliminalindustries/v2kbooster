#! /usr/bin/env python

import sys
import argparse

import v2kbooster

parser = argparse.ArgumentParser(prog='v2kbooster',
                                 description='Boost v2k recordings to an intelligible level')

parser.add_argument('pattern', type=str, help='Glob pattern (ex.: "./*.wav")')
parser.add_argument('--fftn', type=int, help='FFT N', default=8192)

args = parser.parse_args(sys.argv[1:])

v2kbooster.process(args.pattern, nfft=args.fftn)
