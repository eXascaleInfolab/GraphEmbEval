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

from scipy.spatial.distance import squareform, pdist, cdist
# from scipy.sparse import dok_matrix, coo_matrix
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
	"""Load network embeddings from the specified file in the NVC format

	nvcfile: str  - file name

	return
		embeds: matrix  - embeddings matrix in the Compressed Sparse Column format
		dimws: array  - dimensions weights or None
	"""
	hdr = False  # Whether the header is parsed
	ftr = False # Whether the footer is parsed
	ndsnum = 0  # The number of nodes
	dimnum = 0  # The number of dimensions (reprsentative clusters)
	numbered = False
	dimws = None  # Dimension weights
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
	dimens = []  # Dimensions array for the CLUSTER encoding
	nodes = []  # Nodes array for the non CLUSTER encodings

	for ln in nvcfile:
		if not ln:
			continue
		if ln[0] == '#':
			if not hdr:
				# Parse the header
				# Consider ',' separator besides the space
				ln = ' '.join(ln[1:].split[','])
				toks = ln.split(None, len(hdrvals) * 2)
				while toks:
					if len(toks) >= 2:
						key = toks[0].lower()
						if key not in hdrvals:
							break
						hdr = True
						hdrvals[key] = toks[1]
						toks = toks[2:]
					else:
						del toks[:]
				if hdr:
					ndsnum = np.uint32(hdrvals.get('nodes:', ndsnum))
					dimnum = np.uint16(hdrvals.get('dimensions:', dimnum))
					numbered = bool(hdrvals.get('numbered:', numbered))
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
			elif not ftr:
				# Parse the footer
				vals = ln[1:].split(None, 1)
				if not vals or vals[0].lower() != 'diminfo>':
					continue
				ftr = True
				if len(vals) <= 1:
					continue
				vals = vals[1].split()
				if vals and vals[0].find(':') != -1:
					# if valfmt == VAL_UINT8 or valfmt == VAL_UINT16:
					# 	dimws = np.array([np.float32(1. / np.uint16(v[v.find(':') + 1:])) for v in vals], dtype = np.float32)
					# else:
					dimws = np.array([np.float32(v[v.find(':') + 1:]) for v in vals], dtype = np.float32)
			continue
		# Parse the body
		if numbered:
			# Omit the cluster or node id prefix of each row
			ln = ln.split('>', 1)[1]
		if compr == COMPR_CLUSTER:
			vals = ln.split()
			if valfmt == VAL_BIT:
				# tuple(ndids, 1)
				dimens.append((np.array(vals, dtype=np.uint32), 1))
			else:
				nids, vals = zip(*[v.split(':') for v in vals])
				if valfmt == VAL_UINT8 or valfmt == VAL_UINT16:
					vals = [np.float32(1./np.uint16(v)) for v in vals]
				else:
					assert valfmt == VAL_FLOAT32, 'Unexpected valfmt'
				dimens.append((np.array(nids, dtype=np.uint32), np.array(vals, dtype=np.float32)))
		else:
			raise NotImplemented('Non CLUSTER conpression type parsing is not implemented yet')
		irow += 1
	assert not dimnum or dimnum == irow, 'The parsed number of dimensions is invalid'
	# dok_matrix((), dtype=np.float32)


def main():
	training_percents_dfl = [0.9]  # [0.1, 0.5, 0.9]

	parser = ArgumentParser("scoring",
							formatter_class=ArgumentDefaultsHelpFormatter,
							conflict_handler='resolve')
	parser.add_argument("--emb", metavar='EMBEDDING', required=True, help='Embeddings file in the .mat or .nvc format')
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
	# 0. Files
	embeddings_file = args.emb

	# 1. Load Embeddings
	# model = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
	if args.emb.endswith('.mat'):
		mat = loadmat(embeddings_file)
		# Map nodes to their features
		features_matrix = mat['embs']
	elif args.emb.endswith('.nvc'):
		features_matrix = loadNvc(args.emb)
	else:
		raise ValueError('Embeddings in the unknown format specified: ' + args.emb)

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
	for ii, train_percent in enumerate(training_percents):
		for jj, shuf in enumerate(shuffles):
			print([ii,jj])
			X, y = shuf

			training_size = int(train_percent * X.shape[0])

			X_train = X[:training_size, :]
			y_train_ = y[:training_size]

			y_train = [[] for x in range(y_train_.shape[0])]


			cy =  y_train_.tocoo()
			for i, j in zip(cy.row, cy.col):
				y_train[i].append(j)

			assert sum(len(l) for l in y_train) == y_train_.nnz

			X_test = X[training_size:, :]
			y_test_ = y[training_size:]

			y_test = [[] for _ in range(y_test_.shape[0])]

			cy =  y_test_.tocoo()
			for i, j in zip(cy.row, cy.col):
				y_test[i].append(j)

			# Classification strategy and similarity matrices
			clf = TopKRanker(SVC(kernel="precomputed",cache_size=4096,probability=True),1)  # TopKRanker(LogisticRegression())
			gram = squareform( 1 - pdist(X_train, 'cosine'));  # jaccard, hamming; wjaccard
			gram_test = 1 - cdist(X_test, X_train, 'cosine');

			clf.fit(gram, y_train_)

			# find out how many labels should be predicted
			top_k_list = [len(l) for l in y_test]
			preds = clf.predict(gram_test, top_k_list)

			# results = {}
			#
			# for average in averages:
			#     results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
			#
			#  all_results[train_percent].append(res)

			for kk,average in enumerate(averages):
				res[jj,ii,kk] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
	res_ave = np.mean(res,0);
	print("micro, macro")
	print(res_ave)

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

	savemat(args.outputfile,mdict={'res':res})

if __name__ == "__main__":
	main()
