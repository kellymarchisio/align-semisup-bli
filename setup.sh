. ./local-settings.sh

################################################################################
# Get Vecmap and alter one file within it.
git clone https://github.com/artetxem/vecmap.git
cp scripts/map_embeddings.py vecmap
mv vecmap vecmap-fork

# Get Moses scripts.
git clone https://github.com/moses-smt/mosesdecoder.git 

################################################################################
# Get data. 
# 
# These are parts of get_data.sh from Vecmap that are necessary for this project.
################################################################################

DATA=data/embs/artetxe-dinu-embs
mkdir -p $DATA

wget -q http://ixa2.si.ehu.es/martetxe/vecmap/en.emb.txt.gz -O "$DATA/en.emb.txt.gz"
wget -q http://ixa2.si.ehu.es/martetxe/vecmap/it.emb.txt.gz -O "$DATA/it.emb.txt.gz"
gunzip "$DATA/en.emb.txt.gz"
gunzip "$DATA/it.emb.txt.gz"
wget -q http://ixa2.si.ehu.es/martetxe/vecmap/de.emb.txt.gz -O "$DATA/de.emb.txt.gz"
wget -q http://ixa2.si.ehu.es/martetxe/vecmap/fi.emb.txt.gz -O "$DATA/fi.emb.txt.gz"
wget -q http://ixa2.si.ehu.es/martetxe/vecmap/es.emb.txt.gz -O "$DATA/es.emb.txt.gz"
gunzip "$DATA/de.emb.txt.gz"
gunzip "$DATA/fi.emb.txt.gz"
gunzip "$DATA/es.emb.txt.gz"

# Make test and dev sets.
DICTS=dicts/artetxe/
mkdir -p $DICTS
wget -q http://ixa2.si.ehu.es/martetxe/vecmap/dictionaries.tar.gz -O "$DICTS/dictionaries.tar.gz"
tar -xzf "$DICTS/dictionaries.tar.gz" -C "$DICTS"
rm -f "$DICTS/dictionaries.tar.gz"
rm $DICTS/*.train.shuf.txt
for langs in en-de en-fi en-it en-es
do
	tail -n 2000 $DICTS/$langs.train.txt > $DICTS/$langs.train.tail2k.txt
done

################################################################################
# Get & Preprocess Europarl data.
CORPORA=data/corpora
cd $CORPORA
TRG=en
for SRC in es de it fi
do
        wget https://www.statmt.org/europarl/v7/$SRC-$TRG.tgz
        tar -xzvf $SRC-$TRG.tgz
        rm $SRC-$TRG.tgz
done
sh preprocess-data-v7.sh
cd $DIR


