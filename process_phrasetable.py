###############################################################################
# Written by Kelly Marchisio, 2021.
###############################################################################

import argparse
from collections import defaultdict
from utils import utils
import math

def phrasetable_to_trns_probs(phrase_table, min_input_prob, word_counts,
        min_count=2, sep='||_||', to_logprob=False, prob_type='fwd'):
    # Read lexical probabilities from Vecmap / Monoses phrase table extraction:
    # Format output from:
    # https://github.com/artetxem/monoses/blob/master/training/induce-phrase-table.py
    probs = defaultdict(dict)
    if to_logprob:
        min_input_prob = math.log(float(min_input_prob))
    for line in phrase_table:
        src_wd, trg_wd, phrase_probs, *others = [
                item.strip() for item in line.split(sep)]
        if (word_counts['src'][src_wd] < min_count or 
                word_counts['trg'][trg_wd] < min_count):
                continue
        fwdprob, invprob, fwdlexprob, invlexprob = phrase_probs.split()
        fwdprob = math.log(float(fwdprob)) if to_logprob else float(fwdprob)
        invprob = math.log(float(invprob)) if to_logprob else float(invprob)
        prob = 0.
        if prob_type == 'fwd':
            prob = fwdprob
        elif prob_type == 'inv':
            prob = invprob
        else:
            prob = (fwdprob + invprob) / 2
        if prob > min_input_prob:
            probs[src_wd][trg_wd] = prob 
    return probs


def main():
    parser = argparse.ArgumentParser(description='Process phrase table for'
            'translation probabilities.')
    parser.add_argument('--corpus', type=str, required=True, help="the input corpus")
    parser.add_argument('--pt-sep', type=str, default='||_||', help="separator " + 
            "for phrase table output file")
    parser.add_argument('--out-sep', type=str, default='||_||', help="separator " + 
            "for output file")
    parser.add_argument('--prob-type', type=str, 
            choices=['fwd', 'inv', 'avg'], default='fwd', 
            help="Which kind of probability to output (fwd, inv, or avg)")
    parser.add_argument('--phrase-table', default=None, metavar='PATH',
        required=True, help='Phrase table with word translation probs.')
    parser.add_argument('--outfile', metavar='PATH', required=True,
            help='Output file.')
    parser.add_argument('--min-output-prob', default=0.1, type=float,
            help='minimum output probability to write to file.')
    parser.add_argument('--min-count', type=int, default=2, 
            help="minimum number of corpus occurrences for a word to be output")
    parser.add_argument('--to-logprob', type=bool, default=False, 
            help='whether to convert probabilities to log probabilities')
    parser.add_argument('--topk', default=-1, type=int,
            help='output top k hypotheses per source word to write to file.' +
            '-1 means write all.') 
    args = parser.parse_args()
    # If arg received 'None' from shell script, assign it to None.
    for arg in vars(args):
        if getattr(args, arg) == 'None':
            setattr(args, arg, None)

    word_counts = utils.count_words(
            open(args.corpus, 'r', encoding='utf-8', errors='surrogateescape')) 
    phrase_table = phrasetable_to_trns_probs(open(
        args.phrase_table, 'r', encoding='utf-8', errors='surrogateescape'), 
        args.min_output_prob, word_counts, args.min_count, args.pt_sep,
        args.to_logprob, args.prob_type)
    utils.write_probs_outfile(phrase_table, args.outfile, word_counts,
            args.min_count, args.topk, sep=args.out_sep)


if __name__ == '__main__':
    main()
