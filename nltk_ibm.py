# NLTK IBM models
# Written by Kelly Marchiiso, Jan 2021.

import argparse
import copy
from collections import defaultdict
from functools import reduce
import logging
from nltk.translate import AlignedSent, ibm1, ibm2 
import pdb
import process_phrasetable
from utils import utils


def read(f_sents, sep='||_||'):
    # Processes paired sentences separated by sep from a file.
    # Returns sentences and vocabulary for both languages.
    sents = []
    word_counts = {'src': defaultdict(int), 'trg': defaultdict(int)}
    with open(f_sents, 'r', encoding='utf-8', errors='surrogateescape') as f:
        for line in f:
            l1_sent, l2_sent = line.split(sep)
            l1_words = l1_sent.strip().split()
            l2_words = l2_sent.strip().split()
            for word in l1_words:
                word_counts['src'][word] += 1
            for word in l2_words:
                word_counts['trg'][word] += 1
            sents.append(AlignedSent(l1_words, l2_words))
    return sents, word_counts


def read_probs_from_phrasetable(min_input_prob, phrase_table, sep='||_||'):
    # Read lexical probabilities from Vecmap / Monoses phrase table extraction:
    # Format output from:
    # https://github.com/artetxem/monoses/blob/master/training/induce-phrase-table.py
    probs = defaultdict(dict)
    for line in phrase_table:
        src_wd, trg_wd, phrase_probs, *others = [
                item.strip() for item in line.split(sep)]
        invprob, invlexprob, fwdprob, fwdlexprob = phrase_probs.split()
        if float(fwdprob) > min_input_prob:
            probs[src_wd][trg_wd] = float(fwdprob)
    return probs


def read_align_probs(filename, sep="||_||"):
    align_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:defaultdict(lambda: 0.0))))
    with open(filename, 'r', encoding='utf-8', errors='surrogateescape') as f:
        for line in f:
            e1, e2, e3, e4, e5 = line.split(sep)
            e1, e2, e3, e4, e5 = int(e1), int(e2), int(e3), int(e4), float(e5)
            align_prob[e1][e2][e3][e4] = e5
    return align_prob


def merge_probdicts(d1, d2):
    d1 = copy.deepcopy(d1)
    # Updates probabilities in d1 if the pair appears in d2.
    for src_wd in d2:
        for trg_wd in d2[src_wd]:
            if d1.get(src_wd) and trg_wd in d1[src_wd]:
                d1[src_wd][trg_wd] = d2[src_wd][trg_wd]
    return d1


def write_align_probs(align_probs, filename):
    f = open(filename, 'w', encoding='utf-8', errors='surrogateescape')
    for e1 in align_probs:
        for e2 in align_probs[e1]:
            for e3 in align_probs[e1][e2]:
                for e4 in align_probs[e1][e2][e3]:
                    f.write('%d ||_|| %d ||_|| %d ||_|| %d ||_|| %0.7f\n' % (
                        e1, e2, e3, e4, align_probs[e1][e2][e3][e4]))
    f.close()


def run(args):
    logging.info('Running IBM1 with parameters:', args)

    if args.ibm_model == 1:
        ibm = ibm1.IBMModel1
    elif args.ibm_model == 2:
        ibm = ibm2.IBMModel2

    sents, word_counts = read(args.sents)

    ###################
    # Initialize translation and alignment tables.

    # Note - may need to use init_prob_dict and update_probs_from_probdict I 
    # wrote previously if input_trns_probs is too big or gets mad because some pairs
    # aren't initialized to a minimum probability. input_trns_probs sets prob dict 
    # to only contain pairs that cooccur in parallel sentences. 
    # For round 0, this translation table will be uniform, but it's not
    # actually a prob dist b/c nltk assigns just a minimum probability, and
    # doesn't normalize.
    init_ibm = ibm(sents, 0)
    starting_translation_table = init_ibm.translation_table
    starting_alignment_table = copy.deepcopy(init_ibm.alignment_table)

    if args.input_trns_probs:
        logging.info('Reading translation probabilities from %s' % 
                args.input_trns_probs)
        input_trns_probs = utils.dict_from_probsfile(
                open(args.input_trns_probs, 'r', encoding='utf-8',
                    errors='surrogateescape'))
        for srcwd in starting_translation_table:
            if input_trns_probs.get(srcwd):
                starting_translation_table[srcwd].update(input_trns_probs[srcwd])
    # TODO: Original implementation doesn't normlize. I may want to switch
    # this to only normalizing the srcwd if I update it from input_trns_probs 
    logging.info('Normalizing Translation Table')
    normed_starting_translation_table = normalize_prob_dict(
            starting_translation_table)

    if args.input_align_probs:
        logging.info('Reading alignment probabilities from %s' %
                args.input_align_probs)
        input_align_probs = read_align_probs(args.input_align_probs)
        starting_alignment_table.update(input_align_probs)
    
    input_prob_tables = {'translation_table': normed_starting_translation_table,
            'alignment_table': starting_alignment_table}

    #################
    # Run IBM.
    ibm_out = ibm(sents, args.iters, input_prob_tables)
    final_probs = ibm_out.translation_table
    align_probs = ibm_out.alignment_table
    # Trim probs that are less than args.min_input_prob.
    for src in final_probs:
        # Source: https://stackoverflow.com/questions/23862406/filter-items-in-a-python-dictionary-where-keys-contain-a-specific-string
        final_probs[src] = {src:trg for (src, trg) in final_probs[src].items()
                if trg > args.min_output_prob}

    #################
    # Write Output Files.
    logging.info('Writing output probabilities to file...')
    utils.write_probs_outfile(final_probs, args.outfile, word_counts, args.min_count, args.topk)
    write_align_probs(align_probs, args.outfile + '.align')
    logging.info('Done')


def normalize_prob_dict(prob_dict):
    prob_dict_copy = copy.deepcopy(prob_dict)
    # Inspiration:
    # https://stackoverflow.com/questions/12229064/mapping-over-values-in-a-python-dictionary
    # https://stackoverflow.com/questions/16417916/normalizing-dictionary-values
    for srcwd in prob_dict:
        total_prob = float(reduce(lambda x, y: x + y,
            prob_dict[srcwd].values()))
        prob_dict_copy[srcwd] = {k: v / total_prob for k,v in
                prob_dict[srcwd].items()}
    return prob_dict_copy


def main():
    parser = argparse.ArgumentParser(description='IBM Model 1')
    parser.add_argument('--sents', metavar='PATH', help='Training sentences')
    parser.add_argument('--ibm_model', type=int, default=2, help='IBM Model.')
    parser.add_argument('--input-trns-probs', default=None, metavar='PATH',
        help='input word translation probs.'
        'If omitted, use training sents.')
    parser.add_argument('--input-align-probs', metavar='PATH', default=None, 
            help='Input alignment probability table')
    parser.add_argument('--outfile', metavar='PATH', required=True,
            help='Output file.')
    parser.add_argument('--min-input-prob', default=0.001, type=float,
            help='minimum input probability to read in.')
    parser.add_argument('--min-output-prob', default=0.1, type=float,
            help='minimum output probability to write to file.')
    parser.add_argument('--min-count', type=int, default=2, 
            help="minimum number of corpus occurrences for a word to be output")
    parser.add_argument('--topk', default=1, type=int,
            help='output top k hypotheses per source word to write to file.' + 
            'Note: -1 gives you all but the least probable per source word,' + 
            'above the threshold.')
    parser.add_argument('--iters', type=int, default=5,
            help='number of iterations to run IBM Model')
    args = parser.parse_args()
    # If arg received 'None' from shell script, assign it to None.
    for arg in vars(args):
        if getattr(args, arg) == 'None':
            setattr(args, arg, None)
    run(args)


if __name__ == '__main__':
    main()
