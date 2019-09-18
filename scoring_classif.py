#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:Description: Graph (network) embedding evaluation script using multi-label classification.
	Based on the DeepWalk scoring.py script of Bryan Perozzi:
	https://github.com/phanein/deepwalk/tree/master/example_graphs
:Authors: Artem Lutov <luart@ya.ru>, Bryan Perozzi, Dingqi Yang
:Organizations: eXascale lab <http://exascale.info/>, Lumais <http://www.lumais.com/>
:Date: 2019-03
"""
from __future__ import print_function, division  # Required for stderr output, must be the first import
from utils.parser_nvc import loadNvc  #pylint: disable=E0611,E0401
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS as arg_SUPPRESS
# from collections import defaultdict
# from gensim.models import Word2Vec, KeyedVectors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.svm import LinearSVC

from scipy.spatial.distance import squareform, pdist, cdist, cosine as dist_cosine, hamming as dist_hamming
# from scipy.sparse import coo_matrix
from scipy.io import loadmat, savemat
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import dok_matrix
from shutil import move

from hashlib import md5

try:
	# External package: pip install future
	from future.utils import viewitems  #pylint: disable=W0611
except ImportError:
	viewitems = lambda dct: viewMethod(dct, 'items')()  #pylint: disable=W0611

import numpy as np
import os
import sys
import time

# Enable optimized routines from the Cython lib
OPTIMIZED = True  # True
# Enable automatic rebuild of the Cython modules on changes
# (might have dependency issues in the end-user environment)
AUTOREBUILD = False  # True; Otherwise: $ python setup.py build_ext --inplace
if OPTIMIZED:
	if AUTOREBUILD:
		import pyximport; pyximport.install()  # Automatic build of the Cython modules
	import similarities as sm

PROFILE = True
if PROFILE:
	try:
		import cProfile as prof
	except ImportError:
		import profile as prof
	import io
	try:
		from pstats import SortKey
		sk_time = SortKey.TIME
		sk_cumulative = SortKey.CUMULATIVE
	except ImportError:
		sk_time = 'time'  # 1  # 'time'
		sk_cumulative = 'cumulative'  # 2, 'cumulative'
	import pstats


# Predefined types
ValT = np.float32


class TopKRanker(OneVsRestClassifier):
	def predict(self, gram_test, top_k_list):
		assert gram_test.shape[0] == len(top_k_list)
		probs = super(TopKRanker, self).predict_proba(gram_test)
		if not isinstance(probs, np.ndarray):
			probs = probs.toarray()
		all_labels = []
		for i, k in enumerate(top_k_list):
			probs_ = probs[i]
			# Fetch test labels
			labels = self.classes_[probs_.argsort()[-k:]].tolist()
			all_labels.append(labels)
		return all_labels


# def sparse2graph(x):
# 	G = defaultdict(lambda: set())
# 	cx = x.tocoo()
# 	for i,j,v in zip(cx.row, cx.col, cx.data):
# 		G[i].add(j)
# 	return {str(k): [str(x) for x in v] for k,v in viewitems(G)}
#
# def kernel_hamming(X, Y):
#	return np.count_nonzero(a==b)/len(X)


# Default values
_trainperc_dfl = [0.3, 0.5, 0.7]  # [0.1, 0.5, 0.9]  # [0.9]


def parseArgs(opts=None):
	"""Arguments parser

	opts: list(str)  - command line arguments

	return args  - parsed arguments
	"""
	parser = ArgumentParser(description='Network embedding evaluation using multi-lable classification.',
							formatter_class=ArgumentDefaultsHelpFormatter,
							conflict_handler='resolve')
	subparsers = parser.add_subparsers(title='Embedding processing modes', dest='mode') #, description='Modes of the input embedding processing')
									   # , help='Embedding processing modes')
	evaluator = subparsers.add_parser('eval', help='Evaluate embedding.')
	gram = subparsers.add_parser('gram', help='Produce Gram (network nodes similarity) matrix.')
	subparsers.add_parser('test', help='Run doc tests for all modules including "similarities".')  # Mote: has no any specific parameters

	# # Allow either tests execution or embedding evaluation
	# egr = parser.add_mutually_exclusive_group(required=True)
	# egr.add_argument("-e", "--embedding", help='File name of the embedding in .mat or .nvc format')  # , required=True
	# egr.add_argument("--run-tests", default=False, action='store_true', help='Run doc tests for all modules including "similarities".')

	parser.add_argument("-w", "--weighted-dims", default=False, action='store_true',
						help='Apply dimension weights if specified (applicable only for the NVC format).')
	parser.add_argument("--no-dissim", default=False, action='store_true',
						help='Omit dissimilarity weighting (if weights are specified at all).')
	parser.add_argument("-d", "--dimensions", default=None, type=int, dest='dims', help='Use only the specified number of dimensions (clusers) bounded with [root_dims, total_dims].'
						' Out of range values are automatically adjusted to the bound. Actual only for the NVC format.')
	parser.add_argument("--root-dims", default=False, dest='rdims', action='store_true', help=arg_SUPPRESS) # Deprecated; 'Use only root (top) level dimensions (clusers), actual only for the NVC format. Same as "--dimensions 1"'
	parser.add_argument("--dim-vmin", default=0, type=float, help='Minimal dimension value to be processed before the weighting, [0, 1).')
	parser.add_argument("-m", "--metric", default='cosine', help='Applied metric for the similarity matrics construction: cosine, jaccard, hamming.')
						# ', jacnop (jaccard normalized probabilistic, requires input features normalized to 1 and works well only for the classification task).'
	parser.add_argument("-b", "--binarize", default=False, action='store_true', help='Binarize the embedding minimizing the Mean Square Error.'
						' NOTE: the median binarizaion is performed if the hamming metric is specified with this flag.')
	parser.add_argument("-o", "--output", default=None, help='A file name for the results. Default: ./<embeds>.res or ./gtam_<embeds>.mat.')
	parser.add_argument("--num-shuffles", default=5, type=int, help='Number of shuffles of the embedding matrix, >= 1.')
	parser.add_argument("-p", "--profile", default=False, action='store_true', help='Profile the application execution.')
	parser.add_argument("--no-cython", default=False, action='store_true', help='Disable optimized routines from the Cython libs.')

	evaluator.add_argument("-e", "--embedding", required=True, help='File name of the embedding in .mat, .nvc or .csv/.ssv (text) format.')  # , required=True
	evaluator.add_argument("-n", "--network", required=True,
						help='An input network (graph): a .mat file containing the adjacency matrix and node labels.')
	evaluator.add_argument("--adj-matrix-name", default='network',
						help='Variable name of the adjacency matrix inside the network .mat file.')
	evaluator.add_argument("--label-matrix-name", default='group',
						help='Variable name of the labels matrix inside the network .mat file.')
	evaluator.add_argument("-s", "--solver", default=None, help='Linear Regression solver: liblinear (fastest), lbfgs (less accurate, slower, parallel).'
						' ATTENTION: has priority over the SVM kernel.')
	evaluator.add_argument("-k", "--kernel", default='precomputed', help='SVM kernel: precomputed (fast but requires gram/similarity matrix),'
						' rbf (accurate, slow), linear (slow).')
	evaluator.add_argument("--balance-classes", default=False, action='store_true', help='Balance (weight) the grouund-truth classes by their size.')
	evaluator.add_argument("--all", default=False, action='store_true',
						help='The embedding is evaluated on all training percents from 10 to 90 when this flag is set to true. '
						'By default, only training percents of {} are used.'.format(', '.join([str(v) for v in _trainperc_dfl])))
	evaluator.add_argument("--num-shuffles", default=5, type=int, help='Number of shuffles of the embedding matrix, >= 1.')
	# parser.add_argument("--gram", default=None, help='Produce Gram (network nodes similarity) matrix in the MAT format from the embedding'
	# 					' instead of the embedding evaluation.')
	# parser.add_argument("-r", "--results", default=None, help='A file name for the aggregated evaluation results. Default: ./<embeds>.res.')
	evaluator.add_argument("--accuracy-detailed", default=False, help='Output also detailed accuracy evalaution results to ./acr_<evalres>.mat.')

	gram.add_argument("-e", "--embedding", required=True, help='File name of the embedding in .mat or .nvc, .nvc or .csv/.ssv (text) format.')  # , required=True

	args = parser.parse_args(opts)
	if args.mode == 'test':  # args.run_tests:
		return args

	assert 0 <= args.dim_vmin < 1, 'dim_vmin is out of range'
	assert args.num_shuffles >= 1, 'num_shuffles is out of range'
	assert args.metric in ('cosine', 'jaccard', 'hamming', 'jacnop'), 'Unexpexted metric'
	if args.weighted_dims and not args.no_dissim and (args.mode != 'eval' or args.kernel != "precomputed"):
		print('WARNING, dimension no_dissim is automatically set since the dissimilarity weighting'
			' can be considered only for the "precomputed" kernel')
		args.no_dissim = True
	if args.mode == 'eval':
		assert args.solver is None or args.solver in ('liblinear', 'lbfgs'), 'Unexpexted solver'
		assert args.kernel in ('precomputed', 'rbf', 'linear'), 'Unexpexted kernel'

	if args.output is None:
		fname = os.path.splitext(os.path.split(args.embedding)[1])[0]
		if args.mode == 'gram':  # Mode gram
			fname = fname.join(('gram_', '.mat'))
		else:  # Mode eval
			fname += '.res'
		args.output = fname
		print('The output results will be saved to: ', args.output)

	return args


def dist_jaccard(u, v):
	"""Weighted Jaccard distance metric"""
	# Evalaute denominator
	res = np.maximum(u, v).sum()
	# Note: if both modules are 0 then sim ~= 0.5^dims ~= 0: pow(0.5, arrsize)
	# Probability of the similarity is 0.5 on each dimension with confidence 0.5 => 0.25
	if res != 0:
		res = np.minimum(u, v).sum() / res
	else:
		res = pow(0.25, u.size)
	return 1 - res


def dist_jacnop(u, v):
	"""Weighted Jaccard Normalized Probabilistic distance metric

	PREREQUISITES: the input vectors are normalized so as max(abs(x)) = 1 for each item x
	
	>>> round(dist_jacnop(np.array([0, 0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32)), 6)
	0.733333
	"""
	# Evalaute denominator
	norm = np.maximum(u, v)
	res = norm.sum()
	# Note: if both modules are 0 then sim ~= 0.5^dims ~= 0: pow(0.5, arrsize)
	# Probability of the similarity is 0.5 on each dimension with confidence 0.5 => 0.25
	if res != 0:
		res = np.multiply(np.minimum(u, v), norm).sum() / res  # Element-wise multiplication to norm = value confidence
	else:
		res = pow(0.25, u.size)
	return 1 - res


def dis_metric(u, v):
	"""Jaccard-like dissimilarity metric"""
	# Evalaute denominator
	res = np.maximum(u, v).sum()
	# Note: if both modules are 0 then sim ~= 0.5^dims ~= 0: pow(0.5, arrsize); dissim ~= 1 - sim
	# Probability of the similarity is 0.5 on each dimension with confidence 0.5 => 0.25
	if res != 0:
		res = np.absolute(u - v).sum() / res
	else:
		res = 1 - pow(0.25, u.size)
	return res


def pairsimdis(features, dis_features, metric, dis_metric):
	"""Evaluate pairwise similarity (Gram) matrix

	features  - features matrix (node embedding vectors)
	dis_features  - features matirx weighted for the dissimilarity
	metric: callable  - applied similarity metric (cosine, jaccard)
	dis_metric: callable  - applied dissimilarity metric

	return  pairwise similarity (Gram) matrix
	"""
	assert features.shape == dis_features.shape, 'Feature matrices shapes are not synced'
	size = features.shape[0]
	sims = np.empty(features.shape[0] * (size - 1) // 2, dtype=ValT)
	icur = 0
	for i in range(size - 1):
		for j in range(i + 1, size):
			#sims[icur] = ValT(1) - metric(X_train[i], X_train[j]) - dis_metric(Xdis_train[i], Xdis_train[j])
			# Note: positive gram matrix yields abit more accurate resutls
			# print('> x[i].T.shape: {} ({}, T: {}), asarr(x[i]).shape: {} ({}), ravel(x[i]).shape: {} ({})'
			# 	.format(X_train[i].T.shape, hex(id(X_train[i])), hex(id(X_train[i].T))
			# 	, np.asarray(X_train[i]).shape, hex(id(np.asarray(X_train[i])))
			# 	, np.ravel(X_train[i]).shape, hex(id(np.ravel(X_train[i]))) ))
			sims[icur] = ValT(1) - (metric(features[i], features[j])
				+ dis_metric(dis_features[i], dis_features[j])) / ValT(2)
			icur += 1
	assert icur == len(sims), 'sims size validation failed'
	return squareform(sims)


def adjustRows(num, *mats, traceTime=False):
	"""Adjust the number of rows (1st dimension) of the specified matrices

	num: uint  - the required number of rows
	mats: iterable(MultiDimArray)  - multidimentional matrices or NumPy arrays with C ordering to be adjusted
	traceTime: bool  - perform time tracing

	return res: bool  - the matrices are reduced or the reduction was not necessary

	>>> mt = np.array(((0, 1, 2), (3, 4, 5), (6, 7, 8)), dtype=np.uint8); adjustRows(2, mt) and \
		(mt == np.array(((0, 1, 2), (3, 4, 5)), dtype=np.uint8)).all()
	True
	"""
	tstart = time.clock()
	reduced = False  # At least one matix is reduced
	for i, mt in enumerate(mats):
		if mt is None:
			continue
		if mt.shape[0] < num:
			raise ValueError('ERROR: the input matrix #{} has less rows ({}) than required ({}).'
				.format(i, mt.shape[0], num))
		nparr = isinstance(mt, np.ndarray)
		if nparr and np.isfortran(mt):
			raise ValueError('ERROR: the input ndarray #{} is Fortran but not C-ordered.'.format(i))
		if mt.shape[0] > num:
			msz = list(mt.shape)
			msz[0] = num
			#try:
			if nparr:
				assert mt.flags['OWNDATA'], 'An array is expected, which owns it\'s data to be resized'
				mt.resize(msz, refcheck=False)
			else:
				mt.resize(msz)
			#except ValueError:  # An error may occure in case of the container (array, matrix) does not own its data
			#	mats[i] = mt[:num, ...]  # Note: *mats is interpreted as a tuble and can't be assigned
			reduced = True
	if traceTime:
		print('The {} matrices reduction to {} rows is performed on {} sec'.format(len(mats), num, int(time.clock() - tstart)))
	return reduced


def evalEmbCls(args):
	"""Evaluate graph/network embedding via the multi-lable classification

	args  - parsed arguments
	"""
	assert args, 'Valid args are expected'

	tstart = time.clock()
	tstampt = time.gmtime()
	rootdims = None  # Indices of the root dimensions
	dimrds = None  # Dimension density ratios relative to the possibly indirect super cluster (dimension), typically >= 1
	dimrws = None  # Dimension density ratios relative to the possibly indirect super cluster (dimension), typically <= 1
	dimwsim = None  # Dimension weights (significance ratios)
	dimwdis = None  # Dimension weights for the dissimilarity
	dimnds = None  # Dimensions members (nodes) number

	if args.mode == 'eval':
		# 1.1 Load labels
		mat = loadmat(args.network)  # Compressed Sparse Column format
		# A = mat[args.adj_matrix_name]
		# graph = sparse2graph(A)
		labels_matrix = mat[args.label_matrix_name]  # csc_matrix
		labels_count = labels_matrix.shape[1]
		mlb = MultiLabelBinarizer(range(labels_count))
		lbnds = labels_matrix.shape[0]  # The number of labeled nodes
	else:
		lbnds = None

	# 1.2 Load Embedding
	# model = KeyedVectors.load_word2vec_format(args.embedding, binary=False)
	dimweighted = False
	dis_features_matrix = None  # Dissimilarity features matrix
	embext = os.path.splitext(args.embedding)[1].lower()
	if embext == '.nvc':
		tld0 = time.clock()
		features_matrix, rootdims, dimrds, dimrws, dimwsim, dimwdis, dimnds = loadNvc(args.embedding)
		tldf = time.clock()
		print('Feature matrix loaded on {} sec'.format(int(tldf - tld0)))
		# Cut loaded data to rootdims if required
		if args.rdims:
			if args.dims is None:
				args.dims = 1  # or anything else <= len(rootdims)
			else:
				raise ValueError('Exclusive options --dimensions and --root-dims are specified')
		if args.dims is not None:  #rdims
			if args.dims < rootdims.size:
				args.dims = rootdims.size
			if args.dims > features_matrix.shape[1]:
				args.dims = features_matrix.shape[1]
			print('Reduction to the {} dimensions E [{}, {}] started at {} sec'
				.format(args.dims, rootdims.size, features_matrix.shape[1], int(tldf)))
			# Cut the features_matrix to args.dims E rootdims .. totaldims
			fm = dok_matrix((features_matrix.shape[0], args.dims), dtype=features_matrix.dtype)
			# First, fill the root dims
			for j, idim in enumerate(rootdims):
				fm[:, j] = features_matrix.getcol(idim)
				# print('> colmat type: {}, shape: {}, attrs: {}'.format(type(colmat), colmat.shape, dir(colmat)))
			resdims = np.empty(args.dims, np.uint16)  # rootdims
			resdims[:rootdims.size] = rootdims
			if args.dims > rootdims.size:
				# Fill remained dimensions with the ones having max density step and not belongning to the root
				if dimnds is not None:
					drds = [(i, d, dimnds[i]) for i, d in enumerate(dimrds)]
					# Sort by increasing density step and then number of nodes
					rdmin = min(dimrds)
					drds.sort(key=lambda x: x[1] + x[2] / features_matrix.shape[0] * rdmin)
				else:
					drds = [(i, d) for i, d in enumerate(dimrds)]
					drds.sort(key=lambda x: x[1])
				# print('drds: ', drds[:5], '..', drds[-5:])
				# print('rootdims: ', [(i, dimnds[i]) for i in rootdims[:5]], '..', [(i, dimnds[i]) for i in rootdims[-5:]])
				droot = set(rootdims)
				for j in range(rootdims.size, args.dims):
					idim = drds.pop()[0]
					while idim in droot:
						idim = drds.pop()[0]
					resdims[j] = idim
					fm[:, j] = features_matrix.getcol(idim)
			rootdims = None
			features_matrix = fm
			del fm
			trd1 = time.clock()
			print('  features_matrix reduction completed within {} sec'.format(int(trd1 - tldf)))
			# Cut the accessory arrays to rootdims
			arrs = [dimrds, dimrws, dimwsim, dimwdis]
			for ia, arr in enumerate(arrs):
				# Omit None arrays
				if arr is None:
					continue
				tarr = np.empty(resdims.size, arr.dtype)
				for i, ir in enumerate(resdims):
					tarr[i] = arr[ir]
				arrs[ia] = tarr
			resdims = None
			print('  reduction of the accessory loaded data completed on {} sec'.format(int(time.clock() - trd1)))
		allnds = features_matrix.shape[0]
		if lbnds and allnds > lbnds and adjustRows(lbnds, features_matrix, True):
			print('WARNING, embedding matrices are reduced to the number of nodes in the labels matrix: {} -> {}'
				.format(allnds, lbnds), file=sys.stderr)
		# Omit dissimilarity weighting if required
		if args.no_dissim:
			dimwdis = None
		dimweighted = args.weighted_dims and dimwsim is not None
		if dimweighted:
			print('Node vectors are corrected with the dimension weights')
			if dimwdis is not None:
				dis_features_matrix = features_matrix.copy()
			w0 = 1E-8  # Zero weight placeholder
			for (i, j), v in features_matrix.items():
				# Note: Weights cutting must be applied before the dimensions significance consideration
				# w0 is used because 0 assignement does not work in the cycle affecting the dictionary size
				features_matrix[i, j] = v * dimwsim[j] if not args.dim_vmin or v >= args.dim_vmin else w0
			if dis_features_matrix is not None:
				for (i, j), v in dis_features_matrix.items():
					dis_features_matrix[i, j] = v * dimwdis[j] if not args.dim_vmin or v >= args.dim_vmin else w0
				dis_features_matrix = dis_features_matrix.toarray() #.todense() # order='C'
				if OPTIMIZED:
					sm.quantify(dis_features_matrix, sm.CMP_LE, w0, 0)
				else:
					np.where(dis_features_matrix > w0, dis_features_matrix, 0)
		features_matrix = features_matrix.toarray() #.todense() # order='C'
		if dimweighted:
			if OPTIMIZED:
				sm.quantify(features_matrix, sm.CMP_LE, w0, 0)
			else:
				np.where(features_matrix > w0, features_matrix, 0)
	else:
		features_matrix = None
		if embext == '.mat':
			mat = loadmat(args.embedding)
			# Map nodes to their features
			features_matrix = np.array(mat['embs'], dtype=np.float32, order='C')
			del mat
		elif embext == '.csv':
			features_matrix = np.loadtxt(args.embedding, dtype=np.float32, delimiter=',')
		else:  # ssv
			# Try to parse the file as space separated values
			features_matrix = np.loadtxt(args.embedding, dtype=np.float32)
			#raise ValueError('Embedding in the unknown format is specified: ' + args.embedding)
		allnds = features_matrix.shape[0]
		# Ensure that the adday can be resized if required (owns it's data ranther than a view)
		if lbnds and allnds > lbnds:
			if isinstance(features_matrix, np.ndarray) and not features_matrix.flags['OWNDATA']:
				features_matrix = features_matrix[:lbnds, ...]
			else:
				reduced = adjustRows(lbnds, features_matrix, True)
				assert reduced, 'features_matrix is expected to be reduced from {} to {} items'.format(allnds, lbnds)
			embname = os.path.splitext(args.embedding)[0]
			# COnsider that .nvc embeddings support multiple options on loading and should be retained
			if embext != '.nvc':
				embrds = embname + '.mat'
				embdir, namext = os.path.split(args.embedding)
				move(args.embedding, ''.join((embdir, '/', 'full_', namext)))
			else:
				embrds = ''.join((embname, '_rds', str(lbnds), '.mat'))
			print('WARNING, features matrix is reduced to the number of nodes in the labels matrix: {} -> {}.'
				  ' Saving the reduced features to the {}...'
				.format(allnds, lbnds, embrds), file=sys.stderr)
			savemat(embrds, mdict={'embs': features_matrix})

	# Cut weights lower dim_vmin if required
	if args.dim_vmin and not dimweighted:
		if OPTIMIZED:
			sm.quantify(features_matrix, sm.CMP_LT, args.dim_vmin, 0)
		else:
			np.where(features_matrix >= args.dim_vmin, features_matrix, 0)

	# Binarize if required in case of hamming distance evaluation
	if args.binarize:
		medbin = args.metric == 'hamming'  # Binarize to the median instead of reducing mean square error
		sm.binarize(features_matrix, medbin)
		if dis_features_matrix is not None:
			sm.binarize(dis_features_matrix, medbin)

	assert args.metric != 'jacnop' or (features_matrix.max() <= 1 and features_matrix.min() >= -1), (
		'Jacnop should be applied only to the features matrix normalized to 1, i.e. max(abs(mat)) = 1')

	# Generate Gram (nodes similarity) matrix only -----------------------------
	if args.mode == 'gram':
		# Note: metric here is distance metric = 1 - sim_metric
		if OPTIMIZED:
			gram = np.empty((features_matrix.shape[0], features_matrix.shape[0]), dtype=ValT)
			metid = sm.sim_id(args.metric)
		else:
			metric = args.metric
			# Explicitly assign jaccard distance because other metrics are implicitly used as distance metrics
			if metric == 'jaccard':
				metric = dist_jaccard
			elif metric == 'jacnop':
				metric = dist_jacnop
				# metric = lambda u, v: 1 - sm.sim_jaccard(u, v)
		if dis_features_matrix is None:
			if OPTIMIZED:
				# Note: pdist takes too much time with custom dist funciton: 1m46 sec for cosine, 40 sec for jaccard vs 8 sec for "cosine"
				sm.pairsim(gram, features_matrix, metid)
				# gram2 = squareform(ValT(1) - pdist(X_train, metric))  # cosine, jaccard, hamming
				# print('Gram:\n', gram, '\nOrig Gram:\n', gram2)
			else:
				gram = squareform(ValT(1) - pdist(features_matrix, metric))  # cosine, jaccard, hamming
		else:
			if OPTIMIZED:
				sm.pairsimdis(gram, features_matrix, dis_features_matrix, metid)
			else:
				if metric == 'cosine':
					metric = dist_cosine
				elif metric == 'hamming':
					metric = dist_hamming
				if OPTIMIZED:
					dis_metric = sm.dissim
				# else:
				# 	dis_metric = metric  # Note: 1-sim metric performs less accurate than the custom dissimilarity metric
				gram = pairsimdis(features_matrix, dis_features_matrix, metric, dis_metric)
		# Save resulting Gram (network nodes similarity) matrix
		savemat(args.output, mdict={'gram': gram})
		return

	# Evaluate Embedding ------------------------------------------------------
	# Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
	# features_matrix = np.asarray([model[str(node)] for node in range(len(graph))])

	# 2. Shuffle, to create train/test groups
	assert labels_matrix.shape[0] == features_matrix.shape[0], 'All evaluating nodes are expected to be labeled'
	shuffles = []
	for x in range(args.num_shuffles):
		if dis_features_matrix is not None:
			shuffles.append(skshuffle(features_matrix, dis_features_matrix, labels_matrix))
		else:
			shuffles.append(skshuffle(features_matrix, labels_matrix))

	# 3. to score each train/test group
	# all_results = defaultdict(list)

	if args.all:
		training_percents = np.asarray(range(1, 10)) * .1
	else:
		training_percents = _trainperc_dfl

	averages = ["micro", "macro"]
	res = np.full([args.num_shuffles, len(training_percents), len(averages)], np.nan, dtype=ValT)
	# for train_percent in training_percents:
	#     for shuf in shuffles:
	Xdis = None
	Xdis_train = None
	res_ave = None  # Average results
	ii = 0
	jj = 0
	try:
		for ii, train_percent in enumerate(training_percents):
			training_size = int(train_percent * features_matrix.shape[0])
			if OPTIMIZED:
				gram = np.empty((training_size, training_size), dtype=ValT)
			gram_test = np.empty((features_matrix.shape[0] - training_size, training_size), dtype=ValT)
			for jj, shuf in enumerate(shuffles):
				print('Training set #{} ({:.1%}), shuffle #{}'.format(ii, train_percent, jj))
				if dis_features_matrix is not None:
					X, Xdis, y = shuf
					#assert len(X) == len(Xdis), 'Feature matrix partitions validation failed'
				else:
					X, y = shuf

				# training_size = int(train_percent * X.shape[0])
				X_train = X[:training_size]
				if dis_features_matrix is not None:
					Xdis_train = Xdis[:training_size]
				y_train_ = y[:training_size]

				X_test = X[training_size:]
				if dis_features_matrix is not None:
					Xdis_test = Xdis[training_size:]
				if OPTIMIZED:
					y_test = sm.colindicesnz(y[training_size:].tocoo())
				else:
					cy = y[training_size:].tocoo()
					y_test = [[] for _ in range(cy.shape[0])]
					for i, j in zip(cy.row, cy.col):
						y_test[i].append(j)
					cy = None

				# find out how many labels should be predicted
				top_k_list = [len(l) for l in y_test]

				# Classification strategy and similarity matrices
				# clf = TopKRanker(SVC(kernel=args.kernel, cache_size=4096, probability=True), 1)  # TopKRanker(LogisticRegression())
				clf = None
				clweight = 'balanced' if args.balance_classes else None
				if args.solver is None:
					clf = TopKRanker(SVC(kernel=args.kernel, cache_size=4096, probability=True, class_weight=clweight, gamma='scale'))  # TopKRanker(LogisticRegression())
				else:
					clf = TopKRanker(LogisticRegression(solver=args.solver, class_weight=clweight, max_iter=512))
				if args.solver is None and args.kernel == 'precomputed':
					# Note: metric here is distance metric = 1 - sim_metric
					if OPTIMIZED:
						metid = sm.sim_id(args.metric)
					else:
						metric = args.metric
						# Explicitly assign jaccard distance because other metrics are implicitly used as distance metrics
						if metric == 'jaccard':
							metric = dist_jaccard
						elif metric == 'jacnop':
							metric = dist_jacnop
							# metric = lambda u, v: 1 - sm.sim_jaccard(u, v)
					if dis_features_matrix is None:
						if OPTIMIZED:
							# Note: pdist takes too much time with custom dist funciton: 1m46 sec for cosine, 40 sec for jaccard vs 8 sec for "cosine"
							sm.pairsim(gram, X_train, metid)
							# gram2 = squareform(ValT(1) - pdist(X_train, metric))  # cosine, jaccard, hamming
							# print('Gram:\n', gram, '\nOrig Gram:\n', gram2)
							sm.pairsim2(gram_test, X_test, X_train, metid)
							# gram_test2 = ValT(1) - cdist(X_test, X_train, metric);
							# print('\n\nGram test:\n', gram_test, '\nOrig Gram test:\n', gram_test2)
						else:
							gram = squareform(ValT(1) - pdist(X_train, metric))  # cosine, jaccard, hamming
							gram_test = ValT(1) - cdist(X_test, X_train, metric);
					else:
						if OPTIMIZED:
							sm.pairsimdis(gram, X_train, Xdis_train, metid)
							sm.pairsimdis2(gram_test, X_test, X_train, Xdis_test, Xdis_train, metid)
						else:
							if metric == 'cosine':
								metric = dist_cosine
							elif metric == 'hamming':
								metric = dist_hamming
							if OPTIMIZED:
								dis_metric = sm.dissim
							# else:
							# 	dis_metric = metric  # Note: 1-sim metric performs less accurate than the custom dissimilarity metric

							gram = pairsimdis(X_train, Xdis_train, metric, dis_metric)
							# gram_test = 1 - cdist(X_test, X_train, metric);
							#gram_test = np.empty((len(X_test), training_size), dtype=ValT)
							for i in range(len(X_test)):
								for j in range(training_size):
									# gram_test[i, j] = ValT(1) - metric(X_test[i], X_train[j]) - dis_metric(Xdis_test[i], Xdis_train[j])
									# Note: positive gram matrix yields abit more accurate resutls
									gram_test[i, j] = ValT(1) - (metric(X_test[i], X_train[j]) + dis_metric(Xdis_test[i], Xdis_train[j])) / ValT(2)
					clf.fit(gram, y_train_)
					preds = clf.predict(gram_test, top_k_list)
				else:
					clf.fit(X_train, y_train_)
					preds = clf.predict(X_test, top_k_list)

				# results = {}
				#
				# for average in averages:
				#     results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
				#
				#  all_results[train_percent].append(res)

				for kk,average in enumerate(averages):
					res[jj,ii,kk] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
	finally:
		res_ave = np.nanmean(res, 0)
		res_std = np.nanstd(res, 0)
		print("F1 [micro macro]:")
		print(res_ave)
		if len(res_ave) >= 2:
			finres = np.nanmean(res_ave, 0)
			finstd = np.nanmean(res_std, 0)
			print("Average:  {:.4F} ({:.4F}), {:.4F}".format(finres[0], finstd[0], finres[1]))
		else:
			finres = res_ave
			finstd = res_std
		if args.output and ii + jj >= 1:  # Output only non-empty results;  np.nansum(res_ave, 0) != 0
			hbrief = np.uint16(0)
			if args.accuracy_detailed:
				# Evaluate 2-byte hash of the input args
				hf = md5()
				hf.update(' '.join(sys.argv).encode())
				for i, b in enumerate(hf.digest()):
					hbrief = hbrief ^ b << (8 if i%2 else 0)
				# Output detailed accuracy results
				dname, fname = os.path.split(args.embedding)
				acrname = ''.join((dname, '/acr_', os.path.splitext(fname)[0], '_', str(hbrief), '.mat'))
				print('The detailed accuracy results are saved to: ', acrname)
				try:
					savemat(acrname, mdict={'res': res})
				except IOError as err:
					print('WARNING, detailed accuracy results saving falied to {}: {}'
						.format(acrname, err), file=sys.stderr)
			with open(args.output, 'a') as fres:
				# Output the Header if required
				if not fres.tell():
					fres.write('Dims\tWgh\tBin\tMetric \tNDs\tDVmin\t F1mic\tF1miSD\t F1mac\t Solver'
						'\tBCl\t ExecTime\t   Folds\t StartTime        \tInpHash\tEmbeds\n')
				# File name of the embedding and Dimensions number
				print('{: >4}\t{: >3d}\t{: >3d}\t'.format(features_matrix.shape[1], args.weighted_dims, args.binarize)
					, file=fres, end='')
				# Similarity Metric, weighting, no-dissim and dim-val-min
				if args.solver is None and args.kernel == 'precomputed':
					print('{: <7}\t{: >3d}\t'.format(args.metric[:7]
						, args.no_dissim), file=fres, end='')
				else:
					print('{: <7}\t{: >3}\t'.format('-', '-'), file=fres, end='')
				# F1 micro and macro (average value)
				print('{:<.4F}\t {:<.4F}\t{:<.4F}\t {:<.4F}\t '.format(
					args.dim_vmin, finres[0], finstd[0], finres[1]), file=fres, end='')
				# Solver and execution time
				print('{: >6}\t{: >3}\t {: >8d}\t'.format(
					(args.kernel if args.solver is None else args.solver)[:6]
					, int(args.balance_classes), int(time.clock() - tstart)), file=fres, end='')
				# Folds and the timestamp
				# Correct folds to show counts instead of indices
				jj += 1
				if jj == args.num_shuffles:
					ii += 1
				print('{: >2}.{:0>2}/{: >2}.{:0>2}\t {}\t'.format(ii, jj, res.shape[1], res.shape[0]
					, time.strftime('%y-%m-%d_%H:%M:%S', tstampt)), file=fres, end='')
				print('{: >7}\t{}\n'.format(str(hbrief) if hbrief else '-'
					, os.path.split(args.embedding)[1]), file=fres, end='')

	# print ('Results, using embedding of dimensionality', X.shape[1])
	# print ('-------------------')
	# for train_percent in sorted(all_results.keys()):
	#   print ('Train percent:', train_percent)
	#   for index, result in enumerate(all_results[train_percent]):
	#     print ('Shuffle #{:d}:   '.format(index + 1), result)
	#   avg_score = defaultdict(float)
	#   for score_dict in all_results[train_percent]:
	#     for metric, score in viewitems(score_dict):
	#       avg_score[metric] += score
	#   for metric in avg_score:
	#     avg_score[metric] /= len(all_results[train_percent])
	#   print ('Average score:', dict(avg_score))
	#   print ('-------------------')

	#return res_ave


if __name__ == "__main__":
	args = None
	pr = None
	args = parseArgs()
	if args.no_cython:
		OPTIMIZED = False
	if args.mode == 'test':  # args.run_tests:
		# Doc tests execution
		import doctest
		# import pyximport; pyximport.install()
		#doctest.testmod()  # Detailed tests output
		flags = doctest.REPORT_NDIFF | doctest.REPORT_ONLY_FIRST_FAILURE
		modules = [sm, sys.modules[__name__]]
		for md in modules:
			failed, total = doctest.testmod(md, optionflags=flags)
			if failed:
				print("Doctest of the module {} FAILED: {} failures out of {} tests".format(md.__name__, failed, total))
			else:
				print('Doctest of the module {} PASSED: {} tests'.format(md.__name__, total))
	else:
		PROFILE = PROFILE and (not args or args.profile)
		if PROFILE:
			pr = prof.Profile()
			pr.enable()
		# res = None
		try:
			#res = evalEmbCls(args)
			evalEmbCls(args)
		finally:
			if pr:
				pr.disable()
				sio = io.StringIO()
				ps = pstats.Stats(pr, stream=sio).sort_stats(sk_cumulative, sk_time)
				ps.print_stats(30)
				if args and args.output:
					# Trace profiling to the terminal
					print(sio.getvalue(),file=sys.stderr)
					# Output profiling to the file
					profname = os.path.splitext(args.output)[0] + '.prof'
					with open(profname, 'a') as fout:
						fout.write('$ ')
						fout.write(' '.join(sys.argv))
						# # Output evaluation results
						# if res is not None:
						# 	fout.write('\nF1 [micro macro]:')
						# 	print(res, file=fout)
						# 	if len(res) >= 2:
						# 		print("Average:", np.nanmean(res, 0), file=fout)
						fout.write('\n')
						fout.write(sio.getvalue())
						#fout.write('\n')
						fout.write('-'*80)
						fout.write('\n')
