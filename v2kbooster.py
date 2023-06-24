#! /usr/bin/env python

import sys
import argparse
import v2kbooster

parser = argparse.ArgumentParser(prog='v2kbooster',
                                 description='Boost v2k recordings to an intelligible level')

parser.add_argument('-f', '--fftn', required=False, type=int, help='Number of FFT bins (ex.: 8192)', default=8192)
parser.add_argument('-w', '--weights', required=False, type=str, nargs='+',
                    help='List of harmonic weights (ex.: 1. .5 .33 .25 .165)', default=[1., .5, .33, .25, .165])
parser.add_argument('pattern', type=str, help='File glob pattern (ex.: "./*.wav")')

args = parser.parse_args(sys.argv[1:])

v2kbooster.process(args.pattern, nfft=args.fftn, weights=args.weights)
