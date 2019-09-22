#!/usr/bin/env python
# -*- coding: utf-8 -*-
## cython: language_level=3str, infer_type=True
## distutils: extra_compile_args=-fopenmp
## distutils: extra_link_args=-fopenmp
"""
:Description: Similarity functions and embedding vectors to the similarity (gram) matrix conversion utils.
:Authors: Artem Lutov <artem@exascale.info>
:Organizations: eXascale lab <http://exascale.info/>, Lumais <http://www.lumais.com/>
:Date: 2019-03
"""
from __future__ import print_function, division  # Required for stderr output, must be the first import
from scipy.sparse import coo_matrix, isspmatrix_coo
import numpy as np  # Used for the doctests

cdef extern from 'math.h':
	# float fminf(float x, float y) nogil
	float fmaxf(float x, float y) nogil
	float fabsf(float x) nogil
	float powf(float base, float exp) nogil
	float sqrtf(float x) nogil
	float log2f(float x) nogil
	float roundf(float x) nogil
cdef extern from "float.h":
	cdef float FLT_MAX
	cdef float FLT_EPSILON
cdef extern from "errno.h":
	cdef int ENOMEM
from libc.math cimport sqrt as c_sqrt  #, fminf, fmaxf, fabsf
from libc.stdlib cimport malloc, free  # calloc
from libc.stdint cimport uint16_t, UINT16_MAX
# from libc.float cimport FLT_MAX
# from libc.stdio cimport printf
# from cython.parallel import prange
# cimport numpy as np
#
# from libc.math cimport abs as c_abs
# cdef extern from "math.h":
#	double sqrt "c_sqrt"(double x)
cimport cython


# Types declarations -----------------------------------------------------------
ctypedef float  ValT  # Value type, np.float32_t
# ctypedef unsigned char  BoolT
# ctypedef fused TValT:
# 	float
# 	unsigned char
# ctypedef fused ValT:
# 	float
# 	double
# ctypedef floating ValT  # The same as follows and causes generation of the sources for each specialization
# ctypedef fused ValT:
# 	np.float32_t
# 	np.float64_t
# ctypedef np.ndarray[ValT]  ValArrayT
# ctypedef np.ndarray[ValT, ndim=2]  ValMatrixT
ctypedef ValT[::1]  ValArrayT  # C-contiguous 1-dimentional memory view (just ValT[:] defines the strided layout)
# ctypedef BoolT[::1]  BoolArrayT  # C-contiguous 1-dimentional memory view (just ValT[:] defines the strided layout)
# ctypedef TValT[::1]  TValArrayT
# ctypedef fused TValArrayT:
# 	ValT[::1]
# 	BoolT[::1]
# ctypedef ValT[:,::1]  ValArrayT  # C-contiguous 1-dimentional memory view (just ValT[:] defines the strided layout)
# ctypedef fused ValArrayT:
# 	ValT[::1]
# 	ValT[:,::1]
# ctypedef const ValT[::1]  ConstValArrayT  # C-contiguous 1-dimentional memory view (just ValT[:] defines the strided layout)
ctypedef ValT[:,::1]  ValMatrixT  # C-contiguous 2-dimentional memory view
# ctypedef BoolT[:,::1]  BoolMatrixT  # C-contiguous 2-dimentional memory view
# ctypedef TValT[:,::1]  TValMatrixT
# ctypedef fused TValMatrixT:
# 	ValT[:,::1]
# 	BoolT[:,::1]
# ctypedef const ValT[:,::1]  ConstValMatrixT  # C-contiguous 2-dimentional memory view

# Similarity function pointer
ctypedef ValT (*SimilarityF)(ValArrayT a, ValArrayT b) nogil
# ctypedef ValT (*BoolSimilarityF)(BoolArrayT a, BoolArrayT b) nogil
# ctypedef ValT (*TSimilarityF)(TValArrayT a, TValArrayT b) nogil
# ctypedef fused TSimilarityF:
# 	ValT (*)(ValArrayT a, ValArrayT b) nogil
# 	ValT (*)(BoolArrayT a, BoolArrayT b) nogil

# Comparison function pointer
ctypedef bint (*CmpF)(ValT a, ValT b) nogil

# Enum of similarity functions
cpdef enum Similarity:
	SIM_COSINE = 1
	SIM_JACCARD = 2
	SIM_HAMMING = 3
	SIM_JACNOP = 4
	SIM_DISSIM = 0xff

# Enum of comparison functions
cpdef enum Comparison:
	CMP_NE = 0
	CMP_EQ = 1
	CMP_LT = 2
	CMP_LE = 3
	CMP_GT = 4
	CMP_GE = 5

# # Global constants
# cdef ValT  valNaN
# valNaN = float("NaN")

# Function definitions ---------------------------------------------------------
def sim_id(str sim):
	"""Fetch similarity function id by the string name

	sim: str  - name of the similarity funciton

	return simid: Similarity  - id of the similarity function
	"""
	sim = sim.lower()
	if sim == 'cosine':
		return SIM_COSINE
	elif sim == 'jaccard':
		return SIM_JACCARD
	elif sim == 'hamming':
		return SIM_HAMMING
	elif sim == 'jacnop':
		return SIM_JACNOP
	# elif sim == 'dissim':
	# 	return SIM_DISSIM
	else:
		raise ValueError('Unknown similarity literal: ' + sim)


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
def colindicesnz(mat not None):
	"""Form iterable of column indices of the non-zero items per each row

	mat: coo_matrix  - input matrix in the COOrdinate format

	return  res: list  - list of column indices of the non-zero items per each row

	>>> colindicesnz(coo_matrix([[1,0,2], [0,2,0]], dtype=np.uint8))
	[[0, 2], [1]]
	"""
	assert mat.ndim == 2 and isspmatrix_coo(mat), 'A valid COO matrix is expected'
	cdef:
		list  res = [[] for _ in range(mat.shape[0])]
		unsigned  i, r

	for i, r in enumerate(mat.row):
		res[r].append(mat.col[i])
	return res


cdef:
	bint c_ne(ValT a, ValT b) nogil:
		return a != b

	bint c_eq(ValT a, ValT b) nogil:
		return a == b

	bint c_lt(ValT a, ValT b) nogil:
		return a < b

	bint c_le(ValT a, ValT b) nogil:
		return a <= b

	bint c_gt(ValT a, ValT b) nogil:
		return a > b

	bint c_ge(ValT a, ValT b) nogil:
		return a >= b


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
# cdef void c_binarize(BoolMatrixT res, ValMatrixT mat, float eps=1e-4) nogil:
cdef int c_binarize_median(ValMatrixT mat, bint polarize=False, float eps=1e-4) nogil:
	"""Binarize (quantify) matrix values to the median

	mat: ValMatrixT  - the matrix to be binarized
	polarize: bint  - binarize to {-1, 1} instead of {0, 1}
	eps: float  - desirable accuracy error
	"""
	# res: BoolMatrixT  - resulting binarized matrix
	cdef unsigned  i, j, rows = mat.shape[0], cols = mat.shape[1], qsize = <unsigned>(1 / eps)
	cdef unsigned  hqsize = <unsigned>roundf(qsize / 2.), imed, nvals, nv  # Py_ssize_t
	cdef ValT  v, vmin, vmax
	cdef float  dmax
	cdef uint16_t *vdens = <uint16_t*>malloc(sizeof(uint16_t) * qsize)  # calloc(qsize, sizeof(float))

	if not vdens:  # same as 'is NULL' above
		# raise MemoryError('Memory allocation failed for the vdens array')
		return ENOMEM
	try:
		for i in range(rows):  # prange(arrsize, nogil=True))
			# Identify the column values range
			vmin = FLT_MAX
			vmax = -FLT_MAX
			for j in range(cols):
				v = mat[i, j]
				if v < vmin:
					vmin = v
				elif v > vmax:
					vmax = v
			dmax = vmax - vmin
			# Clear the vdens array
			for j in range(qsize):
				vdens[j] = 0
			# Accumulate the number of quantified values to find the median
			if dmax:
				for j in range(cols):
					vdens[<unsigned>roundf((mat[i, j] - vmin) / dmax)] += 1
			else:
				vdens[0] = cols
			# Identify index of the median
			imed = 0  # Index of the median
			nvals = 0  # The accumulated number of the processed values
			for j in range(qsize):
				nv = vdens[j]  # The number of values in the current cell
				if nvals + nv >= hqsize:
					if hqsize - nvals > nvals + nv - hqsize:
						nvals += nv
						imed = j
					break
				if nv:
					nvals += nv
					imed = j
			v = imed * dmax + vmin  # The median value
			# Guarantee that the lowest value is below the binarization margin
			if v - FLT_EPSILON < vmin:
				v += FLT_EPSILON
			# 	# printf('r%u dbmarg: %G (%G <- %G); bmarg p/n: %G/-%G\n', i, fabsf(bmarg - bmargpr), bmarg, bmargpr, pmsqr, bmargpr)
			# Form the resulting binarized mattrix
			for j in range(cols):
				mat[i, j] = mat[i, j] >= v
			# Replace 0 with -1 if polarization is requried
			if polarize:
				for j in range(cols):
					if not mat[i, j]:
						mat[i, j] = -1
	finally:
		free(vdens)
	return 0


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
# cdef void c_binarize(BoolMatrixT res, ValMatrixT mat, float eps=1e-4) nogil:
cdef void c_binarize(ValMatrixT mat, bint polarize=False, float eps=1e-4) nogil:
	"""Binarize (quantify) matrix values minimizing the mean square error

	mat: ValMatrixT  - the matrix to be binarized
	polarize: bint  - binarize to {-1, 1} instead of {0, 1}
	eps: float  - desirable accuracy error
	"""
	# res: BoolMatrixT  - resulting binarized matrix
	cdef unsigned  i, j, ncorr, ncorrmax = <unsigned>log2f(1 / eps) + 1, rows = mat.shape[0], cols = mat.shape[1]  # Py_ssize_t
	cdef ValT  v, vmin, vmax
	cdef float  avg, bmarg, pmsqr, nmsqr, bmargpr, corr
	cdef unsigned  pnum, nnum
	cdef bint  nneg

	for i in range(rows):  # prange(arrsize, nogil=True))
		avg = 0
		vmin = FLT_MAX
		vmax = -FLT_MAX
		for j in range(cols):
			v = mat[i, j]
			avg += v
			if v < vmin:
				vmin = v
			elif v > vmax:
				vmax = v
		avg /= cols
		bmargpr = avg  # bmarg of the previous iteration
		bmarg = (2 * avg + vmin + vmax) / 4.  # Initial binarization margin
		# Adjust bmarg minimizing the mean square error (for v > 1 otherwise mean [linear] error)
		# printf('r%u dbmarg: %G (%G <- %G)\n', i, fabsf(bmarg - bmargpr), bmarg, bmargpr)
		corr = bmarg - bmargpr  # The binary margin correction on the currect iteration
		ncorr = 0  # The number of performed corrections (iterations)
		while fabsf(corr) > eps and ncorr < ncorrmax:
			pmsqr = 0  # Positive mean square error
			pnum = 0
			nmsqr = 0  # Negative mean square error
			nnum = 0
			for j in range(cols):
				v = mat[i, j] - bmarg
				nneg = v >= 0
				v = fabsf(v)
				if v > 1:  # Note: a^2 < a for a E (0, 1)
					v *= v
				if nneg:
					pmsqr += v
					pnum += 1
				else:
					nmsqr += v
					nnum += 1
			if pnum:
				pmsqr /= pnum
				if pmsqr > 1:
					pmsqr = sqrtf(pmsqr)
			if nnum:
				nmsqr /= nnum
				if nmsqr > 1:
					nmsqr = sqrtf(nmsqr)
			bmargpr = bmarg
			bmarg += (pmsqr - nmsqr) / 2
			# Terminate the cycle if the convergence is not occur
			if bmarg - bmargpr >= corr:
				# # Note: bmarg should always <= bmargpr
				# if bmarg >= bmargpr + eps:
				# 	bmarg = bmargpr
				break
			corr = bmarg - bmargpr
			ncorr += 1
			# printf('r%u dbmarg: %G (%G <- %G); bmarg p/n: %G/-%G\n', i, fabsf(bmarg - bmargpr), bmarg, bmargpr, pmsqr, bmargpr)
		# Form the resulting binarized mattrix
		for j in range(cols):
			mat[i, j] = mat[i, j] >= bmarg
		# Replace 0 with -1 if polarization is requried
		if polarize:
			for j in range(cols):
				if not mat[i, j]:
					mat[i, j] = -1


@cython.initializedcheck(False) # Turn off memoryview initialization check
def binarize(ValMatrixT mat not None, bint median=False, bint polarize=True, float eps=1e-4):
	"""Quantify matrix values satisfying the specified condition

	mat: ValMatrixT  - a matrix to be binarized
	median: bint  - binarize to the median or minimizing mean square error
	polarize: bint  - binarize to {-1, 1} instead of {0, 1}
	eps: float  - desirable accuracy error


	>>> mat = np.array([[0, 0.8, 0.5], [0.2, 0.5, 0]], dtype=np.float32); \
		binarize(mat, median=False, polarize=False); \
		(mat == np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8)).all()
	True
	>>> mat = np.array([[0, 0.8, 0.5], [0.2, 0.5, 0]], dtype=np.float32); \
		binarize(mat, False, True); \
		(mat == np.array([[-1, 1, 1], [-1, 1, -1]], dtype=np.int8)).all()
	True
	>>> mat = np.array([[0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=np.float32); \
		binarize(mat, True, False); \
		(mat == np.array([[0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=np.uint8)).all()
	True
	>>> mat = np.array([[0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=np.float32); \
		binarize(mat, True, True); \
		(mat == np.array([[-1, -1, 1, -1, 1, -1], [-1, -1, 1, -1, 1, 1], [1, 1, 1, 1, 1, -1]], dtype=np.int8)).all()
	True
	"""
	cdef int err
	if mat.ndim != 2 or mat.shape[0] > UINT16_MAX or not (0 < eps < 0.1):
		raise ValueError('Valid input matrices are expected, shape ndim: {}, shape[0]: {} / {}, eps: {}'
			.format(mat.ndim, mat.shape[0], UINT16_MAX, eps))
	if median:
		err = c_binarize_median(mat, polarize, eps)
		if err:
			raise RuntimeError('Binarization by median failed with the error code: ' + str(err))
	else:
		c_binarize(mat, polarize, eps)


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
cdef void c_quantify(ValMatrixT mat, Comparison op, ValT cv, ValT qv=0) nogil:
	"""Quantify matrix values satisfying the specified condition

	mat: ValMatrixT  - a matrix to be modified
	op: Comparison  - comparison operation for the matrix values
	cv: ValT  - comparing control value
	qv: ValT  - quantifying value
	"""
	cdef CmpF opf = NULL
	if op == CMP_NE:
		opf = c_ne
	if op == CMP_EQ:
		opf = c_eq
	elif op == CMP_LT:
		opf = c_lt
	elif op == CMP_LE:
		opf = c_le
	elif op == CMP_GT:
		opf = c_gt
	elif op == CMP_GE:
		opf = c_ge
	# else:
	# 	#raise ValueError('Unknown comparison literal: ' + op)
	# 	return -1

	cdef unsigned  i, j, rows = mat.shape[0], cols = mat.shape[1]  # Py_ssize_t

	for i in range(rows):  # prange(arrsize, nogil=True))
		for j in range(cols):
			if opf(mat[i, j], cv):
				mat[i, j] = qv


@cython.initializedcheck(False) # Turn off memoryview initialization check
def quantify(ValMatrixT mat not None, Comparison op, ValT cv, ValT qv=0):
	"""Quantify matrix values satisfying the specified condition

	mat: ValMatrixT  - a matrix to be modified
	op: Comparison  - comparison operation for the matrix values
	cv: ValT  - comparing control value
	qv: ValT  - quantifying value


	>>> mat = np.array([[0, 0.8, 0.5], [0.2, 0.5, 0]], dtype=np.float32); \
		quantify(mat, CMP_GE, 0.6, 0.1); \
		(mat == np.array([[0, 0.1, 0.5], [0.2, 0.5, 0]], dtype=np.float32)).all()
	True
	"""
	if mat.ndim != 2:
		raise ValueError('A valid input matrix is expected, size mat.shape: ' + str(mat.ndim))
	c_quantify(mat, op, cv, qv)


# Note: "nogil" suffix can be used with parallel elementwise operations dealing
# only with c objects and memory views instead of the Python objects.
# A function using a memoryview does not usually need the GIL.
@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
cdef ValT c_sim_cosine(ValArrayT a, ValArrayT b) nogil:
	"""Cosine similarity function

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Cosine similarity between the input arrays
	"""
	# assert a is not None and b is not None and a.shape[0] == b.shape[0], (  # a != NULL
	# 	'Valid arrays of the equal length are expected')
	cdef:
		double  smul = 0  # Scalar multiplication
		double  moda = 0  # Module of the array a
		double  modb = 0  # Module of the array b
		unsigned  i, arrsize = a.shape[0]  # Py_ssize_t
		ValT  va, vb

	for i in range(arrsize):  # prange(arrsize, nogil=True))
		va = a[i]
		vb = b[i]
		smul += va * vb
		moda += va * va
		modb += vb * vb

	if moda != 0 and modb != 0:
		smul /= c_sqrt(moda * modb)
	else:
		# Note: if both modules are 0 then sim ~= 0.5^dims ~= 0
		# Probability of the similarity is 0.5 on each dimension with confidence 0.5 => 0.25
		smul = 0 if moda != modb else powf(0.25, arrsize)
	return smul


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
def sim_cosine(ValArrayT a not None, ValArrayT b not None):
	"""Cosine similarity function

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Cosine similarity between the input arrays

	>>> round(sim_cosine(np.array([0, 0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32)), 6)
	0.787347
	"""
	if a.ndim != 1 or a.shape[0] != b.shape[0]:
		raise ValueError('Valid arrays of the equal length are expected')
	return c_sim_cosine(a, b)


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
cdef ValT c_sim_jaccardwu(ValArrayT a, ValArrayT b) nogil:
	"""(Weighted Unsigned) Jaccard similarity function for non-ngative floating point numbers

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Jaccard similarity between the input arrays, E [0, 1]
	"""
	# assert a is not None and b is not None and a.shape[0] == b.shape[0], (  # a != NULL
	# 	'Valid arrays of the equal length are expected')
	cdef:
		double  nom = 0  # Nomerator of the (Weighted) Jaccard Index
		double  den = 0  # Denomerator of the (Weighted) Jaccard Index
		unsigned  i, arrsize = a.shape[0]  # Py_ssize_t
		ValT  va, vb

	for i in range(arrsize):  # prange
		va = a[i]
		vb = b[i]
		if va <= vb:
			nom += va
			den += vb
		else:
			nom += vb
			den += va
		# if vb < va:
		# 	va = vb
		# 	vb = a[i]
		# nom += va
		# den += vb
	# Note: if both modules are 0 then sim ~= 0.5^dims ~= 0: powf(0.5, arrsize)
	# Probability of the similarity is 0.5 on each dimension with confidence 0.5 => 0.25
	if den != 0:
		nom /= den
	else:
		nom = powf(0.25, arrsize)
	return nom


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
def sim_jaccardwu(ValArrayT a not None, ValArrayT b not None):
	"""(Weighted Unsigned) Jaccard similarity function for arbitrary foating point numbers

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Jaccard similarity between the input arrays

	>>> round(sim_jaccardwu(np.array([0, 0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32)), 6)
	0.333333
	"""
	if a.ndim != 1 or a.shape[0] != b.shape[0]:
		raise ValueError('Valid arrays of the equal length are expected')
	return c_sim_jaccardwu(a, b)


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
cdef ValT c_sim_jaccard(ValArrayT a, ValArrayT b) nogil:
	"""(Weighted Signed) Jaccard similarity function for arbitrary foating point numbers

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Jaccard similarity between the input arrays, E [-1, 1]
	"""
	# assert a is not None and b is not None and a.shape[0] == b.shape[0], (  # a != NULL
	# 	'Valid arrays of the equal length are expected')
	cdef:
		double  nom = 0  # Nomerator of the (Weighted) Jaccard Index
		double  den = 0  # Denomerator of the (Weighted) Jaccard Index
		unsigned  i, arrsize = a.shape[0]  # Py_ssize_t
		ValT  va, vb
		bint  inv  # The numbers have distinct sign (inverse correlation)

	for i in range(arrsize):  # prange
		va = a[i]
		vb = b[i]
		inv = (va < 0) != (vb < 0)
		va = fabsf(va)
		vb = fabsf(vb)
		if va <= vb:
			nom += va * (1 - 2 * inv)
			den += vb
		else:
			nom += vb * (1 - 2 * inv)
			den += va
		# if vb < va:
		# 	va = vb
		# 	vb = fabsf(a[i])
		# nom += va * (1 - 2 * inv)
		# den += vb
	# Note: if both modules are 0 then sim ~= 0.5^dims ~= 0: powf(0.5, arrsize)
	# Probability of the similarity is 0.5 on each dimension with confidence 0.5 => 0.25
	if den != 0:
		nom /= den
	else:
		nom = powf(0.25, arrsize)
	return nom


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
def sim_jaccard(ValArrayT a not None, ValArrayT b not None):
	"""(Weighted Signed) Jaccard similarity function for arbitrary foating point numbers

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Jaccard similarity between the input arrays

	>>> round(sim_jaccard(np.array([0, 0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32)), 6)
	0.333333
	>>> sim_jaccard(np.array([0, 0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32)) == \
		sim_jaccardwu(np.array([0, 0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32))
	True
	>>> round(sim_jaccard(np.array([0, -0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32)), 6)
	-0.333333
	"""
	if a.ndim != 1 or a.shape[0] != b.shape[0]:
		raise ValueError('Valid arrays of the equal length are expected')
	return c_sim_jaccard(a, b)


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
cdef ValT c_sim_jacnop(ValArrayT a, ValArrayT b) nogil:
	"""(Weighted Signed) Jaccard Normalized Probabilistic similarity function for foating point numbers E [-1, 1]

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Jaccard similarity between the input arrays, E [-1, 1]
	"""
	# assert a is not None and b is not None and a.shape[0] == b.shape[0], (  # a != NULL
	# 	'Valid arrays of the equal length are expected')
	cdef:
		double  nom = 0  # Nomerator of the (Weighted) Jaccard Index
		double  den = 0  # Denomerator of the (Weighted) Jaccard Index
		unsigned  i, arrsize = a.shape[0]  # Py_ssize_t
		ValT  va, vb
		bint  inv  # The numbers have distinct sign (inverse correlation)

	for i in range(arrsize):  # prange
		va = a[i]
		vb = b[i]
		inv = (va < 0) != (vb < 0)
		va = fabsf(va)
		vb = fabsf(vb)
		nom += va * vb * (1 - 2 * inv)
		den += vb if va <= vb else va
	# Note: if both modules are 0 then sim ~= 0.5^dims ~= 0: powf(0.5, arrsize)
	# Probability of the similarity is 0.5 on each dimension with confidence 0.5 => 0.25
	if den != 0:
		nom /= den
	else:
		nom = powf(0.25, arrsize)
	return nom


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
def sim_jacnop(ValArrayT a not None, ValArrayT b not None):
	"""(Weighted Signed) Jaccard Normalized Probabilistic similarity function for foating point numbers E [-1, 1]

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Jaccard similarity between the input arrays

	>>> round(sim_jacnop(np.array([0, 0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32)), 6)
	0.266667
	>>> sim_jacnop(np.array([0, 0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32)) == \
		sim_jacnop(np.array([0.2, 0.5, 0], dtype=np.float32), np.array([0, 0.8, 0.5], dtype=np.float32))
	True
	>>> round(sim_jacnop(np.array([0, -0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32)), 6)
	-0.266667
	"""
	if a.ndim != 1 or a.shape[0] != b.shape[0]:
		raise ValueError('Valid arrays of the equal length are expected')
	return c_sim_jacnop(a, b)


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
cdef ValT c_sim_hamming(ValArrayT a, ValArrayT b) nogil:
# cdef ValT c_sim_hamming(BoolArrayT a, BoolArrayT b) nogil:
	"""Hamming similarity function

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Hamming similarity between the input arrays
	"""
	# assert a is not None and b is not None and a.shape[0] == b.shape[0], (  # a != NULL
	# 	'Valid arrays of the equal length are expected')
	cdef:
		unsigned  nom = 0  # Nomerator of the (Weighted) Jaccard Index
		unsigned  i, arrsize = a.shape[0]  # Py_ssize_t

	for i in range(arrsize):  # prange
		# Note: <bint> casting is not performed to guarantee correct evaluation for uint values, i.e. hamming(5, 3) = 0
		nom += a[i] == b[i]
	return <float>nom / arrsize


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
def sim_hamming(ValArrayT a not None, ValArrayT b not None):
	"""(Weighted) Jaccard similarity function

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Hamming similarity between the input arrays

	>>> round(sim_hamming(np.array([0, 1, 1], dtype=np.float32), np.array([1, 1, 0], dtype=np.float32)), 6)
	0.333333
	"""
	if a.ndim != 1 or a.shape[0] != b.shape[0]:
		raise ValueError('Valid arrays of the equal length are expected')
	return c_sim_hamming(a, b)


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
cdef ValT c_dissim(ValArrayT a, ValArrayT b) nogil:
	"""(Weighted) Jaccard-like dissimilarity function

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Jaccard-like dissimilarity between the input arrays
	"""
	# assert a is not None and b is not None and a.shape[0] == b.shape[0], (  # a != NULL
	# 	'Valid arrays of the equal length are expected')
	cdef:
		double  nom = 0  # Nomerator of the (Weighted) Jaccard Index
		double  den = 0  # Denomerator of the (Weighted) Jaccard Index
		unsigned  i, arrsize = a.shape[0]  # Py_ssize_t
		ValT  va, vb

	for i in range(arrsize):  # prange
		va = a[i]
		vb = b[i]
		nom += fabsf(va - vb)
		den += fmaxf(va, vb)
	# Note: if both modules are 0 then sim ~= 0.5^dims ~= 0: powf(0.5, arrsize)
	# Probability of the similarity is 0.5 on each dimension with confidence 0.5 => 0.25
	if den != 0:
		nom /= den
	else:
		nom = powf(0.25, arrsize)
	return nom


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
def dissim(ValArrayT a not None, ValArrayT b not None):
	"""(Weighted) Jaccard-like dissimilarity function

	Preconditions: a is not None and b is not None and a.shape[0] == b.shape[0]

	a: ValArrayT  - first array
	b: ValArrayT  - second array

	return
		sim: ValT  - Jaccard-like dissimilarity between the input arrays

	>>> round(dissim(np.array([0, 0.8, 0.5], dtype=np.float32), np.array([0.2, 0.5, 0], dtype=np.float32)), 6)
	0.666667
	"""
	if a.ndim != 1 or a.shape[0] != b.shape[0]:
		raise ValueError('Valid arrays of the equal length are expected')
	return c_dissim(a, b)


cdef SimilarityF c_sim_metric(Similarity sim):
	"""Fetch similarity metric function pointer by the enum value

	sim: Similarity  - requested similarity function

	return simf: SimilarityF  - the resulting similarity metric funciton pointer
	"""
	if sim == SIM_COSINE:
		return c_sim_cosine
	elif sim == SIM_JACCARD:
		return c_sim_jaccard
	elif sim == SIM_HAMMING:
		return c_sim_hamming
	elif sim == SIM_JACNOP:
		return c_sim_jacnop
	# elif sim == SIM_DISSIM:
	# 	return c_dissim
	else:
		raise ValueError('Unexpected similarity function: ' + str(sim))


# cdef BoolSimilarityF c_boolsim_metric(Similarity sim):
# 	"""Fetch bool similarity metric function pointer by the enum value
#
# 	sim: Similarity  - requested similarity function
#
# 	return simf: BoolSimilarityF  - the resulting similarity metric funciton pointer
# 	"""
# 	if sim == SIM_HAMMING:
# 		return c_sim_hamming
# 	else:
# 		raise ValueError('Unexpected bool similarity function: ' + str(sim))
#
#
# def tsim_metric(Similarity sim):
# 	"""Fetch similarity metric function pointer by the enum value
#
# 	sim: Similarity  - requested similarity function
#
# 	return simf: TSimilarityF  - the resulting similarity metric funciton pointer
# 	"""
# 	if sim == SIM_HAMMING:
# 		return c_sim_hamming
# 	else:
# 		if sim == SIM_COSINE:
# 			return c_sim_cosine
# 		elif sim == SIM_JACCARD:
# 			return c_sim_jaccard
# 		else:
# 			raise ValueError('Unexpected similarity function: ' + str(sim))


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
def pairsim(ValMatrixT res not None, ValMatrixT x not None, Similarity sim):
# def pairsim(ValMatrixT res not None, TValMatrixT x not None, Similarity sim):
# def pairsim(ValMatrixT res not None, TValMatrixT x, TSimilarityF simf):
# def pairsim(ValMatrixT res not None, TValT[:,::1] x, ValT (*simf)(TValT[::1] a, TValT[::1] b) nogil):
	"""Compose pairwise similarity (Gram) matrix for the input array of vectors

	res: ValMatrixT  - resulting similarity matrix NxN. Note: all values are rewritten
	x: ValMatrixT  - input array of vectors NxD
	sim: Similarity  - applied similarity metric

	>>> res = np.empty((2, 2), dtype=np.float32);\
		pairsim(res, np.array([[0, 0.8, 0.5], [0.2, 0.5, 0]], dtype=np.float32), SIM_JACCARD);\
		np.round(res, 6).sum() == np.array([[1, 0.333333], [0.333333, 1]], dtype=np.float32).sum()\
			and res.shape == (2, 2)
	True
	"""
	assert res.shape[0] == res.shape[1] and res.shape[0] == x.shape[0], 'Matrix shapes validation failed'

	cdef:
		SimilarityF  simf = c_sim_metric(sim)
		unsigned  ia, ib, iend = x.shape[0]
		ValT  selfsim = 1 #if sim != SIM_DISSIM else 0

	if iend >= 2:
		# Disable GIL lock
		with nogil:
			for ia in range(iend-1):  # prange(iend-1, nogil=True)
				for ib in range(ia+1, iend):
					res[ia, ib] = simf(x[ia], x[ib])
					res[ib, ia] = res[ia, ib]
				res[ia, ia] = selfsim
	res[iend-1, iend-1] = selfsim


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
def pairsimdis(ValMatrixT res not None, ValMatrixT xs not None, ValMatrixT xd not None, Similarity sim):
	"""Compose pairwise similarity (Gram) matrix for the input arrays of similarity
	and dissimilarity based weighted vectors

	res: ValMatrixT  - resulting similarity matrix NxN. Note: all values are rewritten
	xs: ValMatrixT  - input array of similarity based weighted vectors NxD
	xd: ValMatrixT  - input array of dissimilarity based weighted vectors NxD
	sim: Similarity  - applied similarity metric
	"""
	assert res.shape[0] == res.shape[1] and xs.shape == xd.shape, 'Matrix shapes validation failed'

	cdef:
		SimilarityF  simf = c_sim_metric(sim)
		unsigned  ia, ib, iend = xs.shape[0]
		ValT  selfsim = 1 #if sim != SIM_DISSIM else 0

	if iend >= 2:
		# Disable GIL lock
		with nogil:
			for ia in range(iend-1):  # prange(iend-1, nogil=True)
				for ib in range(ia+1, iend):
					# res[ia, ib] = simf(xs[ia], xs[ib]) - c_dissim(xd[ia], xd[ib])  # Note: signed values provide lower accuracy
					res[ia, ib] = (simf(xs[ia], xs[ib]) - c_dissim(xd[ia], xd[ib]) + 1) / 2
					res[ib, ia] = res[ia, ib]
				res[ia, ia] = selfsim
	res[iend-1, iend-1] = selfsim


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
def pairsim2(ValMatrixT res, ValMatrixT xa not None, ValMatrixT xb not None, Similarity sim):
	"""Compose pairwise similarity (Gram) matrix for each inter pair of the vectors of the input arrays

	res: ValMatrixT  - resulting similarity matrix NxM. Note: all values are rewritten
	xa: ValMatrixT  - input array of vectors NxD
	xb: ValMatrixT  - input array of vectors MxD
	sim: Similarity  - applied similarity metric

	>>> res = np.empty((2, 1), dtype=np.float32);\
		pairsim2(res, np.array([[0, 0.8, 0.5], [0.2, 0.5, 0]], dtype=np.float32)\
			, np.array([[0.3, 0.6, 0]], dtype=np.float32), SIM_JACCARD);\
		np.round(res, 6).sum() == np.array([[0.375], [0.777778]], dtype=np.float32).sum()\
			and res.shape == (2, 1)
	True
	"""
	assert (res.shape[0] == xa.shape[0] and res.shape[1] == xb.shape[0]
		and xa.shape[1] == xb.shape[1]), 'Matrix shapes validation failed'

	cdef:
		SimilarityF  simf = c_sim_metric(sim)
		unsigned  ia, ib

	# Disable GIL lock
	with nogil:
		for ia in range(xa.shape[0]):  # prange(xa.shape[0], nogil=True)
			for ib in range(xb.shape[0]):
				res[ia, ib] = simf(xa[ia], xb[ib])


@cython.boundscheck(False) # Turn off bounds-checking for entire function
@cython.wraparound(False) # Turn off negative index wrapping for entire function
@cython.initializedcheck(False) # Turn off memoryview initialization check
def pairsimdis2(ValMatrixT res, ValMatrixT xas not None, ValMatrixT xbs not None
, ValMatrixT xad not None, ValMatrixT xbd not None, Similarity sim):
	"""Compose pairwise similarity (Gram) matrix for each inter pair of the vectors
	of the input arrays of similarity and dissimilarity based weighted

	res: ValMatrixT  - resulting similarity matrix NxM. Note: all values are rewritten
	xas: ValMatrixT  - input array of similarity based weighted vectors NxD
	xad: ValMatrixT  - input array of dissimilarity based weighted vectors NxD
	xbs: ValMatrixT  - input array of similarity based weighted vectors MxD
	xbb: ValMatrixT  - input array of dissimilarity based weighted vectors MxD
	sim: Similarity  - applied similarity metric
	"""
	assert (res.shape[0] == xas.shape[0] and res.shape[1] == xbs.shape[0]
		and xas.shape[1] == xbs.shape[1] and xas.shape == xad.shape
		and xbs.shape == xbd.shape), 'Matrix shapes validation failed'

	cdef:
		SimilarityF  simf = c_sim_metric(sim)
		unsigned  ia, ib

	# Disable GIL lock
	with nogil:
		for ia in range(xas.shape[0]):  # prange(xa.shape[0], nogil=True)
			for ib in range(xbs.shape[0]):
				# res[ia, ib] = simf(xas[ia], xbs[ib]) - c_dissim(xad[ia], xbd[ib])  # Note: signed values provide lower accuracy
				res[ia, ib] = (simf(xas[ia], xbs[ib]) - c_dissim(xad[ia], xbd[ib]) + 1) / 2
