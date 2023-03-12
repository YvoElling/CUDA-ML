import argparse

parser = argparse.ArgumentParser(prog='CUDA-ML')
parser.add_argument('-mm', '--matrix', action="store_true")
parser.add_argument('-lr', '--regression', action="store_true")
parser.add_argument('-d', '--dense', action="store_true")

args = parser.parse_args()