#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:Description: Networks converter from the mathlab format to the nsl
:Authors: Artem Lutov <luart@ya.ru>
:Organizations: eXascale lab <http://exascale.info/>, Lumais <http://www.lumais.com/>
:Date: 2019-02
"""
from __future__ import print_function, division  # Required for stderr output, must be the first import
from scipy.io import loadmat
import os  # Pathes processing
import argparse
import sys


def matToNsl(mnet, dirnet=None, outdir=None, backup=True):
	"""Convert unsigned input network in the Mathlab .mat format to .nsl

	mnet: str  - unsigned input network in the mathlab format
	dirnet: bool  - the input network can be directed (adjacency matrix can be asymmetric)
		and directed output network should be produced (.nsa format instead of .nse)
	outdir: str  - output directory. Default: directory of the mnet file
	backup: bool  - backup the existing output file
	"""
	print('Converting the {}directed network {}...'.format('' if dirnet else 'un', mnet))
	matnet = loadmat(mnet)
	nc = matnet['network'].tocoo()  # Fetch the network data and convert to the COOrdinate format
	assert nc.col.size == nc.row.size and nc.shape[0] == nc.shape[1], (
		'The adjacency matrix is expected to be square')
	# Omit weight in the unweighted matrix considering that selgweight might be counted twice for the directed networks
	wnds = 0  # The number of self links (weighed nodes, diagonal values != 0)
	wndsWeight = 0.  # Weight of the weighted nodes
	totWeight = 0.
	sys.stdout.write('  Weighted nodes: ')  # print('  Weighted nodes: ', end='')
	for i, j, w in zip(nc.row, nc.col, nc.data):
		totWeight += w
		if i == j:
			wnds += 1
			wndsWeight += w
	print(' {{{}}}'.format(wnds))
	# dsum = nc.data.sum()
	if totWeight == nc.data.size or (wnds * 2 == wndsWeight
	and totWeight == nc.data.size + wnds):
		nc.data = None

	# Create the destination file backing up the existing one if any
	netname = os.path.splitext(mnet)[0]
	if outdir is not None:
		netname = '/'.join((outdir, os.path.split(netname)[1]))
	netext = '.nsa' if dirnet else '.nse'
	onet = netname + netext
	if backup and os.path.isfile(onet):
		onetbk = onet + '.bck'
		# On Windows delete the existing file if required
		if os.path.isfile(onetbk):
			os.remove(onetbk)
		os.rename(onet, onetbk)
	# Write destination header
	# # NSL[A,E] format specification:
	# # Comments are marked with '#' symbol and allowed only in the begin of the line
	# # Optional Header as a special comment:
	# # [Nodes: <nodes_num>[,]	<Links>: <links_num>[,] [Weighted: {0, 1}]]
	# # Note: the comma is either always present as a delimiter or always absent
	# # Body, links_num lines (not considering comments that might be present)
	# <src_id> <dst_id> [<weight>]
	# ...
	# # where:
	# #  nodes_num  - the number of nodes (vertices) in the network (graph)
	# #  Links  - are either Edges (for the undirected network) or Arcs (for the directed network)
	# #  links_num  - the number of <links> (edges or arcs) in the network
	# #  weighted = {0, 1}  - whether the network is weighted or not, default: 1
	# #
	# #  src_id  - source node id >= 0
	# #  dst_id  - destination node id >= 0
	# #  weight  - weight in case the network is weighted, non-negative floating point number
	dirnet = dirnet or bool((nc.col.size - wnds) % 2)  # Force the network processing as directed
	if nc.col.size % 2:
		print('WARNING, an undirected network without selflinks is expected.'
			' Arcs: {} / {} ({} selfarcs). The network is {}directed.'
			.format(nc.col.size - wnds, nc.col.size, wnds, '' if dirnet else 'un'))
	else:
		assert (nc.col.size - wnds) % 2 == 0, 'The number of arcs in undirected network should be even'
	with open(onet, 'w') as fout:
		# Note: use ' Arcs' to have the same number of symbols as in the 'Edges';
		# weighted nodes may increase the number of links not more than by one digit (=> the space is reserved)
		hdr = '# Nodes: {}\t{}: {} \tWeighted: {}\n'.format(nc.shape[0], ' Arcs' if dirnet else 'Edges',
			nc.col.size if dirnet else (nc.col.size - wnds) // 2 + wnds, int(nc.data is not None))
		fout.write(hdr)
		# Write the body
		# Note: the network initially specified in the Compresed Sparse Column format (indexed by columns)
		if nc.data is None:
			for i, j in zip(nc.col, nc.row):
				if dirnet or i <= j:
					fout.write('{} {}\n'.format(i, j))
		else:
			for i, j, w in zip(nc.col, nc.row, nc.data):
				if dirnet or i <= j:
					fout.write('{} {} {}\n'.format(i, j, w))
	print('  Converted to:', onet)


def parseArgs(params=None):
	"""Parse input parameters (arguments)

	params  - the list of arguments to be parsed (argstr.split()), sys.argv is used if args is None

	return
		directed: bool  - the input networks can be directed (the adjacency matrix can be asymmetric)
			and directed output networks should be produced (.nsa format instead of .nse)
		nets: list(str)  - input networks to be converted
		path_outp: str  - output directory
	"""
	parser = argparse.ArgumentParser(description='Network converter from mathlab format to .nsl (nse/nsa).')
	parser.add_argument('mnets', metavar='MatNet', type=str, nargs='+', help='Unsigned input network(s) in the .mat format')
	parser.add_argument('-d', '--directed', dest='directed', action='store_true'
		, help='form directed output network from possibly directed input network')
	parser.add_argument('-p', '--path-outp', default=None, help='Path (directory) for the output files.'
		' Default: respective directory of the input file')
	args = parser.parse_args(params)
	return args.directed, args.mnets, args.path_outp


if __name__ == '__main__':
	dirnet, mnets, outdir = parseArgs()
	for mnet in mnets:
		try:
			matToNsl(mnet, dirnet, outdir)
		except OSError as err:
			print('  conversion of {} failed: {}', mnet, err)
