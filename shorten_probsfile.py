#########################################
# Written by Kelly Marchisio, 2021.
#########################################

from utils import utils
import sys

input_pt_probs_file = sys.argv[1]
outfile = sys.argv[2]
corpus = sys.argv[3]
mincount = int(sys.argv[4])
maxcount = int(sys.argv[5])

word_counts = utils.count_words(open(corpus, 'r', encoding='utf-8', 
    errors='surrogateescape'))
input_pt_probs = utils.dict_from_probsfile(open(input_pt_probs_file, 'r', 
    encoding='utf-8', errors='surrogateescape'))
utils.write_probs_outfile(input_pt_probs, outfile, word_counts, mincount, None,
        maxcount=maxcount)

