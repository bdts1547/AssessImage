from argparse import Namespace
import argparse

args = Namespace(a=1, b='c')
args.d = '3'


parser = argparse.ArgumentParser()

parser.add_argument('action', type=str, default='inference', help='Model Training or Testing options')
parser.add_argument('aaa', type=str, default='inference', help='Model Training or Testing options')
parser.add_argument('acdfion', type=str, default='inference', help='Model Training or Testing options')


cfg = parser.parse_args()


print(args)


