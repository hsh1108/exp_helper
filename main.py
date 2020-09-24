import os, sys, glob
import time
import logging
import torch

from tools import utils
from configs import args_parser

def main(args):
    # define experiment settings
    args.savedir = os.path.join(args.savedir, 'exp-{}'.format(time.strftime("%Y%m%d-%H%M%S")))

    utils.mkdir(args.savedir)
    # define logger
	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
						format=log_format, datefmt='%Y/%m/%d %I:%M:%S %p')
	fh = logging.FileHandler(os.path.join(args.savedir, 'log.txt'))
	fh.setFormatter(logging.Formatter(log_format))
	logger = logging.getLogger()
	logger.addHandler(fh)

if __name__ == '__main__':
    args = args_parser()
	main(args)
