#!/bin/bash
#!/usr/bin/env sh
#
# \description  Parallel execution of the benchmarking
#
# \author Artem V L <artem@exascale.info>  https://exascale.info
#
# Note: Uses the benchmarking is based on the GNU Parallel, paper:
# O. Tange (2011): GNU Parallel - The Command-Line Power Tool, USENIX Magazine, February 2011:42-47.

FREEMEM="8G"  # 8G+ for youtube; 5%
OUTP="res/algs.res"
GRAMDIR="res/gram"
##METRIC=cosine  # cosine, jaccard, hamming
METRICS="cosine jaccard"  # "cosine jaccard hamming"  # jacnop
METRMARK=([cosine]=c [jaccard]=j [hamming]=h [jacnop]=p)  # Note: spaces are significant
#METRICS=cosine
EMBDIMS=128
CLSDIMS=''  # Disable
#ROOTDIMS=''
BINARIZE=''
# Algorithms based on the cosine similarity metric-based optimization dimensions building
ALGORITHMS="Deepwalk GraRep harp-deepwalk harp-line HOPE LINE12 netmf Node2vec Verse"  # NetHash, nodesketch SK_ANH sketch_o1 daoc-gr:=1
# nodesketch, SK_ANH, sketch_o1: uint16 (0 .. 2^16) !!
# Algorithms based on the hamming distance metric-based optimization dimensions building
#ALGORITHMS_HAMMING="LTH_INHMF LTH_ITQ LTH_SGH LTH_SH NetHash, nodesketch, SK_ANH, sketch_o1"
# Types> LTH_INHMF: int16 (-1, 1); LTH_ITQ/SGH/SH: uint8(0, 1);
#ALGORITHMS="GraRep"
GRAPHS="blogcatalog dblp homo wiki" #  youtube
# blogcatalog=blog, wiki=pos, homo=ppi
#GRAPHS=blogcatalog
GRAM=0  # The number of instances for the GRAM matrices evaluation
# Max swappiness, should be 1..10 (low swappiness to hold most of data in RAM)
MAX_SWAP=5

USAGE="$0 -a | [-f <min_available_RAM>] [-o <output>=res/algs.res] [-m \"{`echo $METRICS | tr ' ' ','`} \"+] [-a \"{`echo $ALGORITHMS | tr ' ' ','`} \"+] [-g \"{`echo $GRAPHS | tr ' ' ','`} \"+] [--gram <number>] [-e <embdims>=${EMBDIMS}] [--cls-dims <number>=${CLSDIMS}]
  -f,--free-mem  -  limit the minimal amount of the available RAM to start subsequent job. Default: $FREEMEM
  -o,--output  - output file for the aggregated results on evaluation, execution logs are stored in the same dir. Default: $OUTP
  -m,--metrics  - metrics used for the gram matrix construction. Default: \"$METRICS\"
  -b,--binarize  - binarize embedding by the mean square error
  -a,--algorithms  - evaluationg algorithms. Default: \"$ALGORITHMS\"
  -g,--graphs  - input graphs (networks) specified by the adjacency matrix in the .mat format. Default: \"$GRAPHS\"
  --gram  - evaluate gram matrices for the specified number of embeddings instead of the embeddings accuracy, 0 disables the gram mode
  -e,--emb-dims  - the number of dimensions in the input embeddings or any identifier to select the embeddings filename and load from the dir embs<dims>. Default: $EMBDIMS
  --force-dims  - force the number of cluster-based dimensions to the specified --emb-dims bounding with [root-cls, total-cls]. The out of range value is adjusted to the actual bound, so <=1 means output only the root clusters as dimensions. Actual only for the NVC format
    
  Examples:
  \$ $0 -a \"daoc-g=1 daoc-gr:=1\" --force-dims --gram 5
  \$ $0 -f 8.5G -o res/algs.res -m cosine -a Deepwalk -g 'dblp wiki'
  \$ $0 -m jaccard -a 'daoc-gr:=1' --force-dims -e 128 -g 'blogcatalog dblp homo wiki' --gram 
"
#  -d,--default  - execute everithing with default arguments
#  --root-dims  - evaluate embedding only for the root dimensions (clusters), actual only for the NVC format, same as '--cls-dims 0'

if [ `cat /proc/sys/vm/swappiness` -gt $MAX_SWAP ]
then
	echo "Setting vm.swappiness to $MAX_SWAP..."
	sudo sysctl -w vm.swappiness=$MAX_SWAP
fi

if [ "$LC_ALL" = '' ]  # Note: "" = '' => True
then
	export LC_ALL="en_US.UTF-8"
	export LC_CTYPE="en_US.UTF-8"
	export LANGUAGE="en_US.UTF-8"
fi

if [ $# -lt 1 ]; then
	echo -e "Usage: $USAGE"  # -e to interpret correctly '\n'
	exit 1
fi

while [ $1 ]; do
	case $1 in
	-d|--default)
		# Use defaults for the remained parameters
		break
		;;
	-f|--free-mem)
		if [ "${2::1}" = "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		FREEMEM=$2
		echo "Set $1: $2"
		shift 2
		;;
	-o|--output)
		if [ "${2::1}" = "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		OUTP=$2
		echo "Set $1: $2"
		shift 2
		;;
	-m|--metrics)
		if [ "${2::1}" = "-" ]; then
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
		if [ "${2::1}" = "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		ALGORITHMS=$2
		echo "Set $1: $2"
		shift 2
		;;
	-g|--graphs)
		if [ "${2::1}" = "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		GRAPHS=$2
		echo "Set $1: $2"
		shift 2
		;;
	--gram)
		if [ "${2::1}" = "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		GRAM=$2
		if [ $GRAM -lt 0 ]; then
			echo "ERROR, invalid argument value of $1: $2 (positive integer is expected)"
			exit 1			
		fi
		echo "Set $1: $2"  # GRAPHS are used only to identify embedding file names
		shift 2
		;;
	-e|--emb-dims)
		if [ "${2::1}" = "-" ]; then
			echo "ERROR, invalid argument value of $1: $2"
			exit 1
		fi
		EMBDIMS=$2
		echo "Set $1: $2"
		shift 2
		;;
	--force-dims)
		CLSDIMS=1
		echo "Set $1"
		shift
		;;
#	 --root-dims)
#		CLSDIMS=0
#		ROOTDIMS=$1
#		echo "Set ROOTDIMS: $ROOTDIMS"
#		shift
#		;;
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
mkdir -p "$OUTDIR"
#EXECLOG="$(echo "$OUTP" | cut -f 1 -d '.').log"  # Get first file name in the directory
EXECLOG="${OUTDIR}/algs.log"
echo "ALGORITHMS: $ALGORITHMS"
echo "GRAPHS: $GRAPHS"
echo "EMBDIMS: $EMBDIMS"
#echo "CLSDIMS: $CLSDIMS"
echo "EXECLOG: $EXECLOG"

# Check exictence of the requirements
EXECUTOR=python3  # pypy3
EXECUTORX="$EXECUTOR -O"  # Executor with options
UTILS="free sed bc parallel ${EXECUTOR}"  # awk
for UT in $UTILS; do
	$UT --version
	ERR=$?
	if [ $ERR -ne 0 ]; then
		echo "ERROR, $UT utility is required to be installed, errcode: $ERR"
		exit $ERR
	fi
done

if [ "${FREEMEM:(-1)}" = "%" ]; then
	# Remove the percent sign and evaluate the absolute value from the available RAM
	#FREEMEM=${FREEMEM/%%/}
	#FREEMEM=${FREEMEM::-1}
	#FREEMEM=`free | awk '/^Mem:/{print $2"*1/100"}' | bc` # - total amount of memory (1%); 10G
	FREEMEM=`free | sed -rn "s;^Mem:\s+([0-9]+).*;\1*${FREEMEM::-1}/100;p" | bc`
fi
echo "FREEMEM: $FREEMEM"

# Set CLSDIMS to EMBDIMS if required
SUFDIMS=''
if [ "$CLSDIMS" != "" ]; then
	CLSDIMS="-d $EMBDIMS"
	SUFDIMS="-d${EMBDIMS}"
fi

#echo "> ALGORITHMS: ${ALGORITHMS}, FREEMEM: $FREEMEM"
# embs_{2}_{1}.*  # *: .mat | .nvc
if [ "$GRAM" -ge "1" ]; then
	GRAMDIR=${GRAMDIR}$EMBDIMS
	echo "GRAMDIR: $GRAMDIR"
	mkdir -p $GRAMDIR

	parallel --header : --results "$OUTDIR" --joblog "$EXECLOG" --bar --plus --tagstring {2}${SUFDIMS}_{1}_{3}_{4} --verbose --noswap --memfree ${FREEMEM} --load 96% ${EXECUTORX} scoring_classif.py -m {3} $BINARIZE $CLSDIMS -o "${GRAMDIR}/gram_{2}${SUFDIMS}-m${METRMARK[{3}]}_{1}{4}.mat" gram --embedding embeds/embs${EMBDIMS}/embs_{2}_{1}{4}.* ::: Graphs ${GRAPHS} ::: algs ${ALGORITHMS} ::: metrics ${METRICS} ::: gram $(seq $GRAM)  # $({1..$GRAM})
else
	parallel --header : --results "$OUTDIR" --joblog "$EXECLOG" --bar --plus --tagstring {2}${SUFDIMS}_{1}_{3} --verbose --noswap --memfree ${FREEMEM} --load 96% ${EXECUTORX} scoring_classif.py -m {3} $BINARIZE $CLSDIMS -o "${OUTP}" eval --embedding embeds/embs${EMBDIMS}/embs_{2}${SUFDIMS}-m${METRMARK[{3}]}_{1}.* --network graphs/{1}.mat ::: Graphs ${GRAPHS} ::: algs ${ALGORITHMS} ::: metrics ${METRICS}
fi
