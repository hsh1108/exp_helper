import os, sys, glob
import time
import logging
import torch

import utils
from configs import args_parser


def main():
	# define experiment settings
	args = args_parser()
	args.savedir = os.path.join(args.savedir, 'exp-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
	utils.create_exp_dir(args.savedir, scripts_to_save=glob.glob('*.sh'))
	args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

	# define logger
	log_format = '%(asctime)s %(message)s'
	logging.basicConfig(stream=sys.stdout, level=logging.INFO,
						format=log_format, datefmt='%Y/%m/%d %I:%M:%S %p')
	fh = logging.FileHandler(os.path.join(args.savedir, 'log.txt'))
	fh.setFormatter(logging.Formatter(log_format))
	logger = logging.getLogger()
	logger.addHandler(fh)

	# load dataset
	if args.dataset ==

	# build global model to train
	model =

	utils.save_checkpoint({
		'epoch': epoch + 1,
		'state_dict': model.state_dict(),
		'best_acc_top1': best_acc_top1,
		'optimizer': optimizer.state_dict(),
	}, is_best, args.save)


if __name__ == '__main__':
	main()
