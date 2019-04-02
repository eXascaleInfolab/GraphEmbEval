#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:Description: Graph (network) embedding evaluation script using multi-label classification.
:Authors: Bryan Perozzi, Dingqi Yang, Artem Lutov <luart@ya.ru>
:Organizations: eXascale lab <http://exascale.info/>, Lumais <http://www.lumais.com/>
:Date: 2019-03
"""
from __future__ import print_function, division  # Required for stderr output, must be the first import
from parser_nvc import loadNvc  #pylint: disable=E0611,E0401
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# from collections import defaultdict
# from gensim.models import Word2Vec, KeyedVectors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.svm import LinearSVC

from scipy.spatial.distance import squareform, pdist, cdist, cosine as dist_cosine
# from scipy.sparse import coo_matrix
from scipy.io import loadmat, savemat
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer

try:
	# External package: pip install future
	from future.utils import viewitems  #pylint: disable=W0611
except ImportError:
	viewitems = lambda dct: viewMethod(dct, 'items')()  #pylint: disable=W0611

import numpy as np
import os
import sys

# Enable optimized routines from the Cython lib
OPTIMIZED = True  # True
# Enable automatic rebuild of the Cython modules on changes
# (might have dependency issues in the end-user environment)
AUTOREBUILD = False  # True;
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
_trainperc_dfl = [0.1, 0.5, 0.9]  # [0.9]


def parseArgs(opts=None):
	"""Arguments parser

	opts: list(str)  - command line arguments

	return args  - parsed arguments
	"""
	parser = ArgumentParser(description='Network embedding evaluation using multi-lable classification',
							formatter_class=ArgumentDefaultsHelpFormatter,
							conflict_handler='resolve')
	parser.add_argument("-g", "--network", required=True,
						help='An input network (graph): a .mat file containing the adjacency matrix and node labels.')
	parser.add_argument("--adj-matrix-name", default='network',
						help='Variable name of the adjacency matrix inside the network .mat file.')
	parser.add_argument("--label-matrix-name", default='group',
						help='Variable name of the labels matrix inside the network .mat file.')
	parser.add_argument("-e", "--emb", metavar='EMBEDDING', required=True, help='Embeddings file in the .mat or .nvc format')
	parser.add_argument("-w", "--weighted-dims", default=False, action='store_true',
						help='Apply dimension weights if specified (applicable for the NVC format only)')
	parser.add_argument("--no-dissim", default=False, action='store_true',
						help='Omit dissimilarity weighting (if weights are specified at all)')
	parser.add_argument("--wdim-min", default=0, type=float, help='Minimal weight of the dimension value to be processed, [0, 1)')
	parser.add_argument("-s", "--solver", default=None, help='Linear Regression solver: liblinear, lbfgs (parallel, less accurate)'
						'. ATTENTION: has priority over the SVM kernel')
	parser.add_argument("-k", "--kernel", default='precomputed', help='SVM kernel: precomputed (fastest but requires gram/similarity matrix)'
						', rbf (accurate but slower), linear')
	parser.add_argument("-m", "--metric", default='cosine', help='Applied metric for the similarity matrics construction: cosine, jaccard, hamming.')
	parser.add_argument("--all", default=False, action='store_true',
						help='The embeddings are evaluated on all training percents from 10 to 90 when this flag is set to true. '
						'By default, only training percents of {} are used.'.format(', '.join([str(v) for v in _trainperc_dfl])))
	parser.add_argument("-o", "--output", default='res.mat', help='Output file name for the resulting classifier.')
	parser.add_argument("--num-shuffles", default=10, type=int, help='Number of shuffles of the embedding matrix.')
	parser.add_argument("-p", "--profile", default=False, action='store_true', help='Profile the application execution.')
	parser.add_argument("--sim-tests", default=False, action='store_true', help='Run doc tests for the similarities module.')
	parser.add_argument("--no-cython", default=False, action='store_true', help='Disable optimized routines from the Cython libs.')

	args = parser.parse_args(opts)
	assert 0 <= args.wdim_min < 1, 'wdim_min is out of range'
	assert args.metric in ('cosine', 'jaccard', 'hamming'), 'Unexpexted metric'
	assert args.solver is None or args.solver in ('liblinear', 'lbfgs'), 'Unexpexted solver'
	assert args.kernel in ('precomputed', 'rbf', 'linear'), 'Unexpexted kernel'
	if args.weighted_dims and args.solver is None and args.kernel != "precomputed":
		print('WARNING, dimension weights are omitted since they can be considered only for the "precomputed" kernel')
		args.weighted_dims = False
	return args


def evalEmbCls(args):
	"""Evaluate graph/network embedding via the multi-lable classification

	args  - parsed arguments
	"""
	assert args, 'Valid args are expected'

	# features_matrix, dimwsim = loadNvc('test_cluster_compr.nvc')
	# print('nvec:\n', features_matrix, '\ndimwsim:\n', dimwsim, '\ndimwdis:\n', dimwdis)
	# if dimwsim is not None:
	# 	print('Node vectors are corrected with the dimension weights')
	# 	for (i, j), v in features_matrix.items():
	# 		features_matrix[i, j] = v * dimwsim[j]
	# print(features_matrix)
	# exit(0)

	# 0. Files
	embeddings_file = args.emb
	rootdims = None  # Indices of the root dimensions
	dimrds = None  # Dimension density ratios relative to the possibly indirect super cluster (dimension), typically >= 1
	dimrws = None  # Dimension density ratios relative to the possibly indirect super cluster (dimension), typically <= 1
	dimwsim = None  # Dimension weights (significance ratios)
	dimwdis = None  # Dimension weights for the dissimilarity

	# 1. Load Embeddings
	# model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
	dimweighted = False
	dis_features_matrix = None  # Dissimilarity features matrix
	if args.emb.lower().endswith('.mat'):
		mat = loadmat(embeddings_file)
		# Map nodes to their features
		features_matrix = mat['embs']
	elif args.emb.lower().endswith('.nvc'):
		features_matrix, rootdims, dimrds, dimrws, dimwsim, dimwdis = loadNvc(args.emb)
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
				features_matrix[i, j] = v * dimwsim[j] if not args.wdim_min or v >= args.wdim_min else w0
			if dis_features_matrix is not None:
				for (i, j), v in dis_features_matrix.items():
					dis_features_matrix[i, j] = v * dimwdis[j] if not args.wdim_min or v >= args.wdim_min else w0
				dis_features_matrix = dis_features_matrix.toarray() #.todense() # order='C'
				np.where(dis_features_matrix > w0, dis_features_matrix, 0)
		features_matrix = features_matrix.toarray() #.todense() # order='C'
		if dimweighted:
			np.where(features_matrix > w0, features_matrix, 0)
	else:
		raise ValueError('Embeddings in the unknown format specified: ' + args.emb)

	# Cut weights lower wdim_min if required
	if args.wdim_min and not dimweighted:
		np.where(features_matrix >= args.wdim_min, features_matrix, 0)

	# 2. Load labels
	mat = loadmat(args.network)  # Compressed Sparse Column format
	# A = mat[args.adj_matrix_name]
	# graph = sparse2graph(A)
	labels_matrix = mat[args.label_matrix_name]  # csc_matrix
	labels_count = labels_matrix.shape[1]
	mlb = MultiLabelBinarizer(range(labels_count))

	# Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
	# features_matrix = np.asarray([model[str(node)] for node in range(len(graph))])

	# 2. Shuffle, to create train/test groups
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
	try:
		for ii, train_percent in enumerate(training_percents):
			training_size = int(train_percent * features_matrix.shape[0])
			if OPTIMIZED:
				gram = np.empty((training_size, training_size), dtype=ValT)
			gram_test = np.empty((features_matrix.shape[0] - training_size, training_size), dtype=ValT)
			for jj, shuf in enumerate(shuffles):
				print('Training set #{} ({}%), shuffle #{}'.format(ii, train_percent*100, jj))
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
					y_test_ = y[training_size:]
					y_test = [[] for _ in range(y_test_.shape[0])]
					for i in range(y_test_.shape[0]):
						for j in range(y_test_.shape[1]):
							if y_test_[i, j]:
								y_test[i].append(j)
					# cy = coo_matrix(y_test_)
					# for i, j in zip(cy.row, cy.col):
					# 	y_test[i].append(j)
					y_test_ = None

				# find out how many labels should be predicted
				top_k_list = [len(l) for l in y_test]

				# Classification strategy and similarity matrices
				# clf = TopKRanker(SVC(kernel=args.kernel, cache_size=4096, probability=True), 1)  # TopKRanker(LogisticRegression())
				clf = None
				if args.solver is None:
					clf = TopKRanker(SVC(kernel=args.kernel, cache_size=4096, probability=True, class_weight='balanced', gamma='scale'))  # TopKRanker(LogisticRegression())
				else:
					clf = TopKRanker(LogisticRegression(solver=args.solver, class_weight='balanced'))
				if args.kernel == "precomputed":
					# Note: metric here is distance metric = 1 - sim_metric
					if OPTIMIZED:
						metid = sm.sim_id(args.metric)
					else:
						metric = args.metric
						if metric == "jaccard":
							metric = lambda u, v: ValT(1) - np.minimum(u, v).sum() / np.maximum(u, v).sum()
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
							if metric == "cosine":
								metric = dist_cosine
							if OPTIMIZED:
								dis_metric = sm.dissim
							else:
								def dis_metric(u, v):
									"""Jaccard-like dissimilarity distance metric"""
									return np.absolute(u - v).sum() / np.maximum(u, v).sum()
								#dis_metric = metric  # Note: 1-sim metric performs less accurate than the custom dissimilarity metric

							sims = np.empty(training_size * (training_size - 1) // 2, dtype=ValT)
							icur = 0
							for i in range(training_size - 1):
								for j in range(i + 1, training_size):
									#sims[icur] = ValT(1) - metric(X_train[i], X_train[j]) - dis_metric(Xdis_train[i], Xdis_train[j])
									# Note: positive gram matrix yields abit more accurate resutls
									# print('> x[i].T.shape: {} ({}, T: {}), asarr(x[i]).shape: {} ({}), ravel(x[i]).shape: {} ({})'
									# 	.format(X_train[i].T.shape, hex(id(X_train[i])), hex(id(X_train[i].T))
									# 	, np.asarray(X_train[i]).shape, hex(id(np.asarray(X_train[i])))
									# 	, np.ravel(X_train[i]).shape, hex(id(np.ravel(X_train[i]))) ))
									sims[icur] = ValT(1) - (metric(X_train[i], X_train[j]) + dis_metric(Xdis_train[i], Xdis_train[j])) / ValT(2)
									icur += 1
							assert icur == len(sims), 'sims size validation failed'
							gram = squareform(sims)

							# gram_test = 1 - cdist(X_test, X_train, metric);
							#gram_test = np.empty((len(X_test), training_size), dtype=ValT)
							for i in range(len(X_test)):
								for j in range(training_size):
									# gram_test[i, j] = ValT(1) - metric(X_test[i], X_train[j]) - dis_metric(Xdis_test[i], Xdis_train[j])
									# Note: positive gram matrix yields abit more accurate resutls
									gram_test[i, j] = ValT(1) - (metric(X_test[i], X_train[j]) + dis_metric(Xdis_test[i], Xdis_train[j])) / ValT(2)
					clf.fit(gram, y_train_)
					preds = clf.predict(gram_test, top_k_list)
				else: #if args.solver in ('liblinear', 'lbfgs') or args.kernel in ('rbf', 'linear'):
					clf.fit(X_train, y_train_)
					preds = clf.predict(X_test, top_k_list)
				# else:
				# 	raise ValueError('Unexpected kernel: ' + args.kernel)

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
		print("F1 [micro macro]:")
		print(res_ave)
		if len(res_ave) >= 2:
			print("Average:", np.nanmean(res_ave, 0))

	# print ('Results, using embeddings of dimensionality', X.shape[1])
	# print ('-------------------')
	# for train_percent in sorted(all_results.keys()):
	#   print ('Train percent:', train_percent)
	#   for index, result in enumerate(all_results[train_percent]):
	#     print ('Shuffle #%d:   ' % (index + 1), result)
	#   avg_score = defaultdict(float)
	#   for score_dict in all_results[train_percent]:
	#     for metric, score in viewitems(score_dict):
	#       avg_score[metric] += score
	#   for metric in avg_score:
	#     avg_score[metric] /= len(all_results[train_percent])
	#   print ('Average score:', dict(avg_score))
	#   print ('-------------------')

	savemat(args.output, mdict={'res': res})
	#return res_ave


if __name__ == "__main__":
	args = None
	pr = None
	args = parseArgs()
	if args.no_cython:
		OPTIMIZED = False
	if args.sim_tests:
		# Doc tests execution
		import doctest
		# import pyximport; pyximport.install()
		#doctest.testmod()  # Detailed tests output
		flags = doctest.REPORT_NDIFF | doctest.REPORT_ONLY_FIRST_FAILURE
		failed, total = doctest.testmod(sm, optionflags=flags)
		if failed:
			print("Doctest FAILED: {} failures out of {} tests".format(failed, total))
		else:
			print('Doctest PASSED: {} tests'.format(total))
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
