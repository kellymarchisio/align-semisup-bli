#!/bin/bash

. ./local-settings.sh

SRC=$1
TRG=$2
SIZE=$3
OUTDIR=$4

EVAL_ITER=1

# Target Languages For Artetxe Embs: It, De, Fi, Es
TESTDICT=$DIR/dicts/artetxe/$SRC-$TRG.test.txt
OUTFILE=$OUTDIR/eval.iter$EVAL_ITER

echo Bitext Size: $SIZE
echo Language Pair: $SRC $TRG

python3 $VECMAP/eval_translation.py \
	$OUTDIR/$EVAL_ITER/src.out.txt $OUTDIR/$EVAL_ITER/trg.out.txt \
	-d $TESTDICT --retrieval csls > $OUTFILE
