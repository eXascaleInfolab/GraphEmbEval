#!/bin/bash
#!/bin/sh
#
# \description  Parallel execution of the benchmarking
#
# \author Artem V L <artem@exascale.info>  https://exascale.info
#
# Note: Uses the benchmarking is based on the GNU Parallel, paper:
# O. Tange (2011): GNU Parallel - The Command-Line Power Tool, USENIX Magazine, February 2011:42-47.

FREEMEM="8G"  # 8G+ for youtube; 5%
OUTP="res/algs.res"
##METRIC=cosine  # cosine, jaccard, hamming
METRICS="cosine jaccard"  # "cosine jaccard hamming"
#METRICS=cosine
BINARIZE=''
# Algorithms based on the cosine similarity metric-based optimization dimensions building
ALGORITHMS="Deepwalk GraRep HOPE LINE12 netmf Node2vec NodeSketch SK_ANH sketch_o1 Verse"
# NodeSketch, SK_ANH, sketch_o1: uint16 (0 .. 2^16) !!
# Algorithms based on the hamming distance metric-based optimization dimensions building
#ALGORITHMS_HAMMING="LTH_INHMF LTH_ITQ LTH_SGH LTH_SH"
# Types> LTH_INHMF: int16 (-1, 1); LTH_ITQ/SGH/SH: uint8(0, 1);
#ALGORITHMS="GraRep"
GRAPHS="blogcatalog dblp homo wiki youtube"
#GRAPHS=blogcatalog

USAGE="$0 -a | [-f <min_available_RAM>] [-o <output>=res/algs.res] [-m \"{`echo $METRICS | tr ' ' ','`} \"+] [-a \"{`echo $ALGORITHMS | tr ' ' ','`} \"+] [-g \"{`echo $GRAPHS | tr ' ' ','`} \"+]
  -d,--default  - execute everithing with default arguments
  -f,--free-mem  -  minimal amount of the available RAM to start subsequent job. Default: $FREEMEM
  -o,--output  - results output file. Default: $OUTP
  -m,--metrics  - metrics used for the gram matrix construction. Default: \"$METRICS\"
  -b,--binarize  - binarize embedding by the mean square error
  -a,--algorithms  - evaluationg algorithms. Default: \"$METRICS\"
  -g,--graphs  - input graphs (networks) specified by the adjacency matrix in the .mat format
    
  Examples:
  \$ $0 -d
  \$ $0 -f 8.5G -o res/algs.res -m cosine -a Deepwalk -g 'dblp wiki'
"

if [ $# -lt 1 ]; then
	echo "Usage: $USAGE"
	exit 1
fi

while [ $1 ]; do
	case $1 in
	-d|--default)
		# Use defaults for the remained parameters
		break
		;;
	-f|--free-mem)
		if [ "${2::1}" == "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		FREEMEM=$2
		echo "Set $1: $2"
		shift 2
		;;
	-o|--output)
		if [ "${2::1}" == "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		OUTP=$2
		echo "Set $1: $2"
		shift 2
		;;
	 -m|--metrics)
		if [ "${2::1}" == "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		METRICS=$2
		echo "Set $1: $2"
		shift 2
		;;
	 -b|--binarize)
		BINARIZE=$1
		echo "Set BINARIZE: $BINARIZE"
		shift
		;;
	 -a|--algorithms)
		if [ "${2::1}" == "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		ALGORITHMS=$2
		echo "Set $1: $2"
		shift 2
		;;
	 -g|--graphs)
		if [ "${2::1}" == "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		GRAPHS=$2
		echo "Set $1: $2"
		shift 2
		;;
#	-*)
#		printf "Error: Invalid option specified.\n\n$USAGE"
#		exit 1
#		;;
	*)
		printf "Error: Invalid option specified: $1 $2 ...\n\n$USAGE"
		exit 1
		;;
	esac
done
OUTDIR="$(dirname "$OUTP")"  # Output directory for the executable package
EXECLOG="$(echo "$OUTP" | cut -f 1 -d '.').log"
echo "ALGORITHMS: $ALGORITHMS"
echo "GRAPHS: $GRAPHS"
echo "EXECLOG: $EXECLOG"

# Check exictence of the requirements
EXECUTOR=python3
UTILS="free sed bc parallel ${EXECUTOR}"  # awk
for UT in $UTILS; do
	$UT --version
	ERR=$?
	if [ $ERR -ne 0 ]; then
		echo "ERROR, $UT utility is required to be installed, errcode: $ERR"
		exit $ERR
	fi
done

if [ "${FREEMEM:(-1)}" == "%" ]; then
	# Remove the percent sign and evaluate the absolute value from the available RAM
	#FREEMEM=${FREEMEM/%%/}
	#FREEMEM=${FREEMEM::-1}
	#FREEMEM=`free | awk '/^Mem:/{print $2"*1/100"}' | bc` # - total amount of memory (1%); 10G
	FREEMEM=`free | sed -rn "s;^Mem:\s+([0-9]+).*;\1*${FREEMEM::-1}/100;p" | bc`
fi
echo "FREEMEM: $FREEMEM"

#echo "> ALGORITHMS: ${ALGORITHMS}, FREEMEM: $FREEMEM"
parallel --header : --results "$OUTDIR" --joblog "$EXECLOG" --bar --plus --tagstring {2}_{1}_{3} --verbose --noswap --memfree ${FREEMEM} --load 96% ${EXECUTOR} scoring_classif.py -m {3} ${BINARIZE} -o "${OUTP}" eval --embedding embeds/algsEmbeds/embs_{2}_{1}.mat --network graphs/{1}.mat ::: Graphs ${GRAPHS} ::: Algorithms ${ALGORITHMS} ::: Metrics ${METRICS}
