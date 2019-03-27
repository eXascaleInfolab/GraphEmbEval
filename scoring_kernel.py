#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:Description: Graph (network) embedding evaluation script using multi-label classification.
:Authors: Bryan Perozzi, Dingqi Yang, Artem Lutov <luart@ya.ru>
:Organizations: eXascale lab <http://exascale.info/>, Lumais <http://www.lumais.com/>
:Date: 2019-03
"""
from __future__ import print_function, division  # Required for stderr output, must be the first import
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
# from gensim.models import Word2Vec, KeyedVectors
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.svm import LinearSVC

from scipy.spatial.distance import squareform, pdist, cdist, cosine
from scipy.sparse import dok_matrix  #, coo_matrix
from sklearn.metrics import f1_score
from scipy.io import loadmat, savemat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer

try:
	# External package: pip install future
	from future.utils import viewitems  #pylint: disable=W0611
except ImportError:
	viewitems = lambda dct: viewMethod(dct, 'items')()  #pylint: disable=W0611

import numpy as np
# import sys


class TopKRanker(OneVsRestClassifier):
	def predict(self, gram_test, top_k_list):
		assert gram_test.shape[0] == len(top_k_list)
		probs = np.asarray(super(TopKRanker, self).predict_proba(gram_test))
		all_labels = []
		for i, k in enumerate(top_k_list):
			probs_ = probs[i, :]
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


def loadNvc(nvcfile):
	"""Load network embeddings from the specified file in the NVC format v1.1

	nvcfile: str  - file name

	return
		embeds: matrix  - embeddings matrix in the Dictionary Of Keys sparse matrix format
		dimwsim: array  - dimensions weights for the similarity or None
		dimwdis: array  - dimensions weights for the dissimilarity or None
	"""

	hdr = False  # Whether the header is parsed
	ftr = False # Whether the footer is parsed
	ndsnum = 0  # The number of nodes
	dimnum = 0  # The number of dimensions (reprsentative clusters)
	numbered = False
	dimwsim = None  # Dimension weights for the similarity
	dimwdis = None  # Dimension weights for the dissimilarity
	COMPR_NONE = 0
	COMPR_RLE = 1
	COMPR_SPARSE = 2
	COMPR_CLUSTER = 4  # Default
	compr = COMPR_CLUSTER  # Compression type
	VAL_BIT = 0
	VAL_UINT8 = 1
	VAL_UINT16 = 2
	VAL_FLOAT32 = 4
	valfmt = VAL_UINT8  # Falue format
	hdrvals = {'nodes:': None, 'dimensions:': None, 'value:': None, 'compression:': None, 'numbered:': None}
	irow = 0  # Payload line (matrix row) index (of either dimensions or nodes)
	nvec = None  # Node vectors matrix
	vqnorm = np.uint16(0xFF if valfmt == VAL_UINT8 else 0xFFFF)  # Normalization value for the vector quantification based compression

	with open(nvcfile, 'r') as fnvc:
		for ln in fnvc:
			if not ln:
				continue
			if ln[0] == '#':
				if not hdr:
					# Parse the header
					# Consider ',' separator besides the space
					ln = ' '.join(ln[1:].split(','))
					toks = ln.split(None, len(hdrvals) * 2)
					while toks:
						#print('toks: ', toks)
						key = toks[0].lower()
						isep = 0 if key.endswith(':') else key.find(':') + 1
						if isep:
							val = key[isep:]
							key = key[:isep]
						if key not in hdrvals:
							break
						hdr = True
						if isep:
							hdrvals[key] = val
							toks = toks[1:]
						elif len(toks) >= 2:
							hdrvals[key] = toks[1]
							toks = toks[2:]
						else:
							del toks[:]
					if hdr:
						print
						ndsnum = np.uint32(hdrvals.get('nodes:', ndsnum))
						dimnum = np.uint16(hdrvals.get('dimensions:', dimnum))
						numbered = int(hdrvals.get('numbered:', numbered))  # Note: bool('0') is True (non-empty string)
						comprstr = hdrvals.get('compression:', '').lower()
						if comprstr == 'none':
							compr = COMPR_NONE
						elif comprstr == 'rle':
							compr = COMPR_RLE
						elif comprstr == 'sparse':
							compr = COMPR_SPARSE
						elif comprstr == 'cluster':
							compr = COMPR_CLUSTER
						else:
							raise ValueError('Unknown compression format: ' + compr)
						valstr = hdrvals.get('value:', '').lower()
						if valstr == 'bit':
							valfmt = VAL_BIT
						elif valstr == 'uint8':
							valfmt = VAL_UINT8
						elif valstr == 'uint16':
							valfmt = VAL_UINT16
						elif valstr == 'float32':
							valfmt = VAL_FLOAT32
						else:
							raise ValueError('Unknown value format: ' + valstr)
						#print('hdrvals:',hdrvals, '\nnumbered:', numbered)
				elif not ftr:
					# Parse the footer
					vals = ln[1:].split(None, 1)
					if not vals or vals[0].lower() != 'diminfo>':
						continue
					ftr = True
					if len(vals) <= 1:
						continue
					vals = vals[1].split()
					idimw = vals[0].find(':') + 1
					if vals and idimw:
						# if valfmt == VAL_UINT8 or valfmt == VAL_UINT16:
						# 	dimwsim = np.array([np.float32(1. / np.uint16(v[v.find(':') + 1:])) for v in vals], dtype=np.float32)
						# else:
						if vals[0].find('/', idimw) == -1:
							dimwsim = np.array([np.float32(v[v.find(':') + 1:]) for v in vals], dtype=np.float32)
						else:
							dimwsim = np.array([np.float32(v[v.find(':') + 1:v.rfind('/')]) for v in vals], dtype=np.float32)
							dimwdis = np.array([np.float32(v[v.rfind('/')+1:]) for v in vals], dtype=np.float32)
				continue

			# Construct the matrix
			if not (ndsnum and dimnum):
				raise ValueError('Invalid file header, the number of nodes ({}) and dimensions ({}) should be positive'.format(ndsnum, dimnum))
			# TODO: Ensure that non-float format for bits does not affect the subsequent evaluations or use dtype=np.float32 for all value formats
			if nvec is None:
				nvec = dok_matrix((ndsnum, dimnum), dtype=np.float32 if valfmt != VAL_BIT else np.uint8)

			# Parse the body
			if numbered:
				# Omit the cluster or node id prefix of each row
				ln = ln.split('>', 1)[1]
			vals = ln.split()
			if compr == COMPR_CLUSTER:
				if valfmt == VAL_BIT:
					for nd in vals:
						nvec[np.uint32(nd), irow] = 1
				else:
					nids, vals = zip(*[v.split(':') for v in vals])
					if valfmt == VAL_UINT8 or valfmt == VAL_UINT16:
						# vals = [np.float32(1. / np.uint16(v)) for v in vals]
						vals = [np.float32(vqnorm - np.uint16(v) + 1) / vqnorm for v in vals]
					else:
						assert valfmt == VAL_FLOAT32, 'Unexpected valfmt'
					for i, nd in enumerate(nids):
						nvec[np.uint32(nd), irow] = vals[i]
			elif compr == COMPR_SPARSE:
				if valfmt == VAL_BIT:
					for dm in vals:
						nvec[irow, np.uint32(dm)] = 1
				else:
					dms, vals = zip(*[v.split(':') for v in vals])
					if valfmt == VAL_UINT8 or valfmt == VAL_UINT16:
						# vals = [np.float32(1. / np.uint16(v)) for v in vals]
						vals = [np.float32(vqnorm - np.uint16(v) + 1) / vqnorm for v in vals]
					else:
						assert valfmt == VAL_FLOAT32, 'Unexpected valfmt'
					for i, dm in enumerate(dms):
						nvec[irow, np.uint32(dm)] = vals[i]
			elif compr == COMPR_RLE:
				corr = 0  # RLE caused index correction
				for j, v in enumerate(vals):
					if v[0] != '0':
						if valfmt == VAL_UINT8 or valfmt == VAL_UINT16:
							# nvec[irow, j + corr] = 1. / np.uint16(v)
							nvec[irow, j + corr] = np.float32(vqnorm - np.uint16(v) + 1) / vqnorm
						else:
							assert valfmt == VAL_FLOAT32 or valfmt == VAL_BIT, 'Unexpected valfmt'
							nvec[irow, j + corr] = v
					elif len(v) >= 2:
						if v[1] != ':':
							raise ValueError('Invalid RLE value (":" separator is expected): ' + v);
						corr = np.uint16(v[2:]) + 1  # Length, the number of values to be inserted / skipped
					else:
						corr += 1
			else:
				assert compr == COMPR_NONE, 'Unexpected compression format'
				corr = 0  # 0 caused index correction
				for j, v in enumerate(vals):
					if v == '0':
						corr += 1
						continue
					if valfmt == VAL_UINT8 or valfmt == VAL_UINT16:
						# nvec[irow, j + corr] = 1. / np.uint16(v)
						nvec[irow, j + corr] = np.float32(vqnorm - np.uint16(v) + 1) / vqnorm
					else:
						assert valfmt == VAL_FLOAT32 or valfmt == VAL_BIT, 'Unexpected valfmt'
						nvec[irow, j + corr] = v
			irow += 1

	assert not dimnum or dimnum == irow, 'The parsed number of dimensions is invalid'
	assert len(dimwsim) == len(dimwdis), 'Parsed dimension weights are not synchronized'
	#print('nvec:\n', nvec, '\ndimwsim:\n', dimwsim, '\ndimwdis:\n', dimwdis)
	# Return node vecctors matrix in the Dictionary Of Keys based sparse format and dimension weights
	return nvec, dimwsim, dimwdis  # nvec.tocsc() - Compressed Sparse Column format


def main():
	# features_matrix, dimwsim = loadNvc('test_cluster_compr.nvc')
	# print('nvec:\n', features_matrix, '\ndimwsim:\n', dimwsim, '\ndimwdis:\n', dimwdis)
	# if dimwsim is not None:
	# 	print('Node vectors are corrected with the dimension weights')
	# 	for (i, j), v in features_matrix.items():
	# 		features_matrix[i, j] = v * dimwsim[j]
	# print(features_matrix)
	# exit(0)
	training_percents_dfl = [0.9]  # [0.1, 0.5, 0.9]

	parser = ArgumentParser(description='Network embedding evaluation using multi-lable classification',
							formatter_class=ArgumentDefaultsHelpFormatter,
							conflict_handler='resolve')
	parser.add_argument("--emb", metavar='EMBEDDING', required=True, help='Embeddings file in the .mat or .nvc format')
	parser.add_argument("-w", "--weighted-dims", default=False, action='store_true',
						help='Apply dimension weights if specified (for .nvc format only)')
	parser.add_argument("--wdim-min", default=0, type=float, help='Minimal weight of the dimension value to be processed, [0, 1)')
	parser.add_argument("--kernel", default='precomputed', help='SVM kernel: precomputed, rbf')
	parser.add_argument("--network", required=True,
						help='A .mat file containing the adjacency matrix and node labels of the input network.')
	parser.add_argument("--metric", default='cosine', help='Applied metric for the similarity matrics construction: cosine, jaccard, hamming.')
	parser.add_argument("--adj-matrix-name", default='network',
						help='Variable name of the adjacency matrix inside the .mat file.')
	parser.add_argument("--label-matrix-name", default='group',
						help='Variable name of the labels matrix inside the .mat file.')
	parser.add_argument("--num-shuffles", default=10, type=int, help='Number of shuffles.')
	parser.add_argument("--outputfile", default='res.mat', help='Number of shuffles.')
	parser.add_argument("--all", default=False, action='store_true',
						help='The embeddings are evaluated on all training percents from 10 to 90 when this flag is set to true. '
						'By default, only training percents of {} are used.'.format(', '.join([str(v) for v in training_percents_dfl])))

	args = parser.parse_args()
	assert 0 <= args.wdim_min < 1, 'wdim_min is out of range'
	assert args.metric in ('cosine', 'jaccard', 'hamming'), 'Unexpexted metric'
	assert args.kernel in ('precomputed', 'rbf'), 'Unexpexted kernel'
	if args.kernel != "precomputed":
		print('WARNING, dimension weights are omitted since they can be considered only for the "precomputed" kernel')
		args.weighted_dims = False
	# 0. Files
	embeddings_file = args.emb
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
		features_matrix, dimwsim, dimwdis = loadNvc(args.emb)
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
				dis_features_matrix = dis_features_matrix.todense()
				np.where(dis_features_matrix > w0, dis_features_matrix, 0)
		features_matrix = features_matrix.todense()
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
	labels_matrix = mat[args.label_matrix_name]
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
		training_percents = training_percents_dfl

	averages = ["micro", "macro"]
	res = np.zeros([args.num_shuffles, len(training_percents), len(averages)])
	# for train_percent in training_percents:
	#     for shuf in shuffles:
	Xdis = None
	Xdis_train = None
	for ii, train_percent in enumerate(training_percents):
		for jj, shuf in enumerate(shuffles):
			print([ii, jj])
			if dis_features_matrix is not None:
				X, Xdis, y = shuf
				#assert len(X) == len(Xdis), 'Feature matrix partitions validation failed'
			else:
				X, y = shuf

			training_size = int(train_percent * X.shape[0])

			X_train = X[:training_size, :]
			if dis_features_matrix is not None:
				Xdis_train = Xdis[:training_size, :]
			y_train_ = y[:training_size]

			y_train = [[] for x in range(y_train_.shape[0])]


			cy =  y_train_.tocoo()
			for i, j in zip(cy.row, cy.col):
				y_train[i].append(j)

			assert sum(len(l) for l in y_train) == y_train_.nnz

			X_test = X[training_size:, :]
			if dis_features_matrix is not None:
				Xdis_test = Xdis[training_size:, :]
			y_test_ = y[training_size:]

			y_test = [[] for _ in range(y_test_.shape[0])]

			cy = y_test_.tocoo()
			for i, j in zip(cy.row, cy.col):
				y_test[i].append(j)

			# find out how many labels should be predicted
			top_k_list = [len(l) for l in y_test]

			# Classification strategy and similarity matrices
			clf = TopKRanker(SVC(kernel=args.kernel, cache_size=4096, probability=True, class_weight='balanced', gamma='scale'), 1)  # TopKRanker(LogisticRegression())
			if args.kernel == "precomputed":
				# Note: metric here is distance metric = 1 - sim_metric
				metric = args.metric
				if metric == "jaccard":
					metric = lambda u, v: 1 - np.minimum(u, v).sum() / np.maximum(u, v).sum()
				if dis_features_matrix is None:
					gram = squareform(1 - pdist(X_train, metric))  # cosine, jaccard, hamming
					gram_test = 1 - cdist(X_test, X_train, metric);
				else:
					def dis_metric(u, v):
						"""Jaccard-like dissimilarity distance metric"""
						return np.absolute(u - v).sum() / np.maximum(u, v).sum()

					if metric == "cosine":
						metric = cosine
					sims = np.zeros(training_size * (training_size - 1) // 2, dtype=np.float32)
					icur = 0
					for i in range(training_size - 1):
						for j in range(i + 1, training_size):
							sims[icur] = 1 - metric(X_train[i], X_train[j]) - dis_metric(Xdis_train[i], Xdis_train[j])
							# sims[icur] = 1 - metric(X_train[i], X_train[j]) - (1 - metric(Xdis_train[i], Xdis_train[j]))
							#sims_test[icur] = 1 - metric(X_test[i], X_test[j]) - (1 - metric(Xdis_test[i], Xdis_test[j]))
							icur += 1
					gram = squareform(sims)
					# gram_test = 1 - cdist(X_test, X_train, metric);
					#gram_test = squareform(sims_test)
					gram_test = np.zeros((len(X_test), training_size), dtype=np.float32)
					for i in range(len(X_test)):
						for j in range(training_size):
							gram_test[i, j] = 1 - metric(X_test[i], X_train[j]) - (1 - metric(Xdis_test[i], Xdis_train[j]))
				clf.fit(gram, y_train_)
				preds = clf.predict(gram_test, top_k_list)
			elif args.kernel == "rbf":
				clf.fit(X_train, y_train_)
				preds = clf.predict(X_test, top_k_list)
			else:
				raise ValueError('Unexpected kernel: ' + args.kernel)

			# results = {}
			#
			# for average in averages:
			#     results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
			#
			#  all_results[train_percent].append(res)

			for kk,average in enumerate(averages):
				res[jj,ii,kk] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
	res_ave = np.mean(res, 0)
	print("F1 [micro macro]:")
	print(res_ave)
	if len(res) >= 2:
		print("Average:", np.mean(res_ave, 0))

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

	savemat(args.outputfile, mdict={'res': res})


if __name__ == "__main__":
	main()
