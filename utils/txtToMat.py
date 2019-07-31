#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:Description: Network Embedding Vectors converter from .csv / .ssv (space separate values) text format to .mat
:Authors: Artem Lutov <artem@exascale.info>
:Organizations: eXascale lab <http://exascale.info/>, Lumais <http://www.lumais.com/>
:Date: 2019-07
"""
from __future__ import print_function, division  # Required for stderr output, must be the first import
from scipy.io import savemat
import os  # Pathes processing
import argparse
import numpy as np


def txtToMat(embname, outdir=None, backup=True):
	"""Convert unsigned input network in the Mathlab .mat format to .nsl

	embname: str  - file name of the network embedding vectors
	outdir: str  - output directory. Default: directory of the mnet file
	backup: bool  - backup the existing output file
	"""
	print('Converting the network embedding {}...'.format(embname))
	outname, embext = os.path.splitext(embname)
	embext = embext.lower()
	emb = None
	if embext == '.csv':
		emb = np.loadtxt(embname, dtype=np.float32, delimiter=',')
	else:  # ssv
		# Try to parse the file as space separated values
		emb = np.loadtxt(embname, dtype=np.float32)

	# Create the destination file backing up the existing one if any
	outname, embext = os.path.splitext(embname)
	if outdir is not None:
		outname = '/'.join((outdir, os.path.split(outname)[1]))
	outname += '.mat'
	if backup and os.path.isfile(outname):
		namebk = outname + '.bck'
		# On Windows delete the existing file if required
		if os.path.isfile(namebk):
			os.remove(namebk)
		os.rename(outname, namebk)

	savemat(outname, mdict={'embs': emb})


def parseArgs(params=None):
	"""Parse input parameters (arguments)

	params  - the list of arguments to be parsed (argstr.split()), sys.argv is used if args is None

	return
		embs: list(str)  - input embeddings to be converted
		outp_dir: str  - output directory
	"""
	parser = argparse.ArgumentParser(description='Network Embedding Vectors converter'
		' from .csv / .ssv (space separate values) text format to .mat.')
	parser.add_argument('-d', '--outp-dir', default=None, help='Path (directory) for the output files')
	args = parser.parse_args(params)
	return args.mnets, args.outp_dir


if __name__ == '__main__':
	embs, outdir = parseArgs()
	for fname in embs:
		try:
			txtToMat(fname, outdir)
		except OSError as err:
			print('  conversion of {} failed: {}', fname, err)