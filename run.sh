#!/bin/bash
#$ -cwd

###############################################################################
#
# Main run script for 
#  "An Alignment-Based Approach to Semi-Supervised Bilingual Lexicon Induction"
#
# Code written by Kelly Marchisio (2020, 2021)
###############################################################################

DIR="$(dirname "$(realpath "$0")")"
. $DIR/local-settings.sh

SRC_EMBS=$1
TRG_EMBS=$2
INPUT_SENTS=$3
OUTDIR=$4
DEVICE=$5

CUDA=
if [ $DEVICE == 'gpu' ]; then CUDA=--cuda; fi

mkdir -p $OUTDIR
echo Outdir: $OUTDIR

MIN_IBM_INPUT_PROB=0.1
MIN_IBM_OUTPUT_PROB=0.1
MIN_COUNT=2
TOPK_OUT_VECMAP=1
MIN_N_FOR_SUP=1000
IBM_N=3000
VECMAP_N=10000

################################################################################

echo Beginning Training 

mkdir -p $OUTDIR/0

# Run IBM Model 2 over input corpus.
python3 $DIR/nltk_ibm.py --sents $INPUT_SENTS --min-input-prob $MIN_IBM_INPUT_PROB \
		--min-output-prob $MIN_IBM_OUTPUT_PROB \
		--outfile $OUTDIR/0/align_out_all 
# Reduce IBM output probs to top $IBM_N. 
cat $OUTDIR/0/align_out_all | sed 's/ ||_|| /\t/g' | sort  -k3r | \
	awk '{print $1 " ||_|| " $2 " ||_|| " $3}' | head -$IBM_N > \
	$OUTDIR/0/align_out

# Map embeddings given seeds from IBM 2 and induce/process phrasetable.
SRC_EMBS_OUT=$OUTDIR/0/src.out.txt
TRG_EMBS_OUT=$OUTDIR/0/trg.out.txt
align_len=`echo \`wc $OUTDIR/0/align_out_all\` | cut -f1 -d' '`
echo alignment length: $align_len
if [ $align_len -lt $MIN_N_FOR_SUP ]; then
    echo Mapping in semi-supervised mode.
    python3 $VECMAP/map_embeddings.py $SRC_EMBS $TRG_EMBS \
    	$SRC_EMBS_OUT $TRG_EMBS_OUT --max_embs 200000 \
    	--sep "||_||" -v --semi_supervised $OUTDIR/0/align_out_all $CUDA 
else
    echo Mapping in supervised mode.
    python3 $VECMAP/map_embeddings.py $SRC_EMBS $TRG_EMBS \
    	$SRC_EMBS_OUT $TRG_EMBS_OUT --max_embs 200000 \
    	--sep "||_||" -v --supervised $OUTDIR/0/align_out_all $CUDA 
fi
python3 $SCRIPTS/induce-phrase-table.py \
	--src $SRC_EMBS_OUT --trg $TRG_EMBS_OUT \
	--src2trg $OUTDIR/0/src2trg.phrase-table \
	--epochs 3 --sep "||_||" --size $TOPK_OUT_VECMAP $CUDA
python3 $DIR/process_phrasetable.py --corpus $INPUT_SENTS \
	--phrase-table $OUTDIR/0/src2trg.phrase-table \
	--outfile $OUTDIR/0/phrasetable_out_all --topk $TOPK_OUT_VECMAP \
        --min-output-prob $MIN_IBM_INPUT_PROB --min-count 0

# Discard words that occurred frequently in the corpus.
python3 $DIR/shorten_probsfile.py $OUTDIR/0/phrasetable_out_all \
	$OUTDIR/0/phrasetable_out_infrequent $INPUT_SENTS 0 $MIN_COUNT
# Reduce Vecmap output probs to top $VECMAP_N infrequent words. 
cat $OUTDIR/0/phrasetable_out_infrequent | sed 's/ ||_|| /\t/g' | \
        sort  -k3r -t$'\t' | awk -F$'\t' '{print $1 " ||_|| " $2 " ||_|| " $3}' | \
        head -$VECMAP_N > $OUTDIR/0/phrasetable_out

mkdir -p $OUTDIR/1
# Join IBM and Vecmap top translations.
VECMAP_INPUT_TRNS_PROBS=$OUTDIR/1/align0_and_pt0_out
cat $OUTDIR/0/align_out $OUTDIR/0/phrasetable_out > $VECMAP_INPUT_TRNS_PROBS

SRC_EMBS_OUT=$OUTDIR/1/src.out.txt
TRG_EMBS_OUT=$OUTDIR/1/trg.out.txt
python3 $VECMAP/map_embeddings.py $SRC_EMBS $TRG_EMBS \
	$SRC_EMBS_OUT $TRG_EMBS_OUT --max_embs 200000 \
	--sep "||_||" -v --supervised $VECMAP_INPUT_TRNS_PROBS $CUDA
