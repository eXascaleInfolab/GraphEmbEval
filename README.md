`\authors` (c) Artem Lutov <artem@exascale.info>  
`\license` Apache 2.0, optional commercial support and relicensing is provided by the request  

# GraphEmbeddingEval
Graph (network) embeddings evaluation sript via the classification.  
This is a significantly modified and extended version of the Python scoring script from the [DeepWalk](https://github.com/phanein/deepwalk/). The extensions include classification using not only the linear regression but also various SVM/SVC kernels, some preprocessing and optimizations implemented using Cython.

Authors (in addition to the authors of the original `DeepWalk`): (c) Artem Lutov <artem@exascale.info>, Dingqi Yang


## Deployment

```
$ ./install_reqs.sh
$ ./build.sh
```

## Usage Examples:
```sh
$ time python3 scoring_classif.py --embeddings embeds/blog.nvc -m cosine -o res/blog.res eval -s liblinear --num-shuffles 3 --network graphs/blog.mat
```

## Options
General Options:
```
$ python3 scoring_classif.py -h
usage: scoring_classif.py [-h] [-w] [--no-dissim] [--root-dims]
                          [--dim-vmin DIM_VMIN] [-m METRIC] [-b] [-o OUTPUT]
                          [--num-shuffles NUM_SHUFFLES] [-p] [--no-cython]
                          {eval,gram,test} ...

Network embedding evaluation using multi-lable classification.

optional arguments:
  -h, --help            show this help message and exit
  -w, --weighted-dims   Apply dimension weights if specified (applicable only
                        for the NVC format). (default: False)
  --no-dissim           Omit dissimilarity weighting (if weights are specified
                        at all). (default: False)
  --root-dims           Use only root (top) level dimensions (clusers), actual
                        only for the NVC format. (default: False)
  --dim-vmin DIM_VMIN   Minimal dimension value to be processed before the
                        weighting, [0, 1). (default: 0)
  -m METRIC, --metric METRIC
                        Applied metric for the similarity matrics
                        construction: cosine, jaccard, hamming. (default:
                        cosine)
  -b, --binarize        Binarize the embedding minimizing the Mean Square
                        Error. NOTE: the median binarizaion is performed if
                        the hamming metric is specified with this flag.
                        (default: False)
  -o OUTPUT, --output OUTPUT
                        A file name for the results. Default: ./<embeds>.res
                        or ./gtam_<embeds>.mat. (default: None)
  --num-shuffles NUM_SHUFFLES
                        Number of shuffles of the embedding matrix, >= 1.
                        (default: 5)
  -p, --profile         Profile the application execution. (default: False)
  --no-cython           Disable optimized routines from the Cython libs.
                        (default: False)

Embedding processing modes:
  {eval,gram,test}
    eval                Evaluate embedding.
    gram                Produce Gram (network nodes similarity) matrix.
    test                Run doc tests for all modules including
                        "similarities".
```

Evaluation Options:
```
$ python3 scoring_classif.py eval -h
usage: scoring_classif.py eval [-h] -e EMBEDDING -n NETWORK
                               [--adj-matrix-name ADJ_MATRIX_NAME]
                               [--label-matrix-name LABEL_MATRIX_NAME]
                               [-s SOLVER] [-k KERNEL] [--balance-classes]
                               [--all] [--num-shuffles NUM_SHUFFLES]
                               [--accuracy-detailed ACCURACY_DETAILED]

optional arguments:
  -h, --help            show this help message and exit
  -e EMBEDDING, --embedding EMBEDDING
                        File name of the embedding in .mat, .nvc or .csv/.ssv
                        (text) format.
  -n NETWORK, --network NETWORK
                        An input network (graph): a .mat file containing the
                        adjacency matrix and node labels.
  --adj-matrix-name ADJ_MATRIX_NAME
                        Variable name of the adjacency matrix inside the
                        network .mat file.
  --label-matrix-name LABEL_MATRIX_NAME
                        Variable name of the labels matrix inside the network
                        .mat file.
  -s SOLVER, --solver SOLVER
                        Linear Regression solver: liblinear (fastest), lbfgs
                        (less accurate, slower, parallel). ATTENTION: has
                        priority over the SVM kernel.
  -k KERNEL, --kernel KERNEL
                        SVM kernel: precomputed (fast but requires
                        gram/similarity matrix), rbf (accurate, slow), linear
                        (slow).
  --balance-classes     Balance (weight) the grouund-truth classes by their
                        size.
  --all                 The embedding is evaluated on all training percents
                        from 10 to 90 when this flag is set to true. By
                        default, only training percents of 0.3, 0.5, 0.7 are
                        used.
  --num-shuffles NUM_SHUFFLES
                        Number of shuffles of the embedding matrix, >= 1.
  --accuracy-detailed ACCURACY_DETAILED
                        Output also detailed accuracy evalaution results to
                        ./acr_<evalres>.mat.
```
