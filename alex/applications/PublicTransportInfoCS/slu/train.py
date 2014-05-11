#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import argparse
import os
from os.path import dirname, exists, join, realpath

if __name__ == "__main__":
    import autopath

from alex.applications.PublicTransportInfoCS.preprocessing import PTICSSLUPreprocessing
from alex.components.asr.utterance import Utterance, UtteranceNBList
from alex.components.slu.da import DialogueAct
from alex.components.slu.base import CategoryLabelDatabase
from alex.components.slu.dailrclassifier import DAILogRegClassifier
from alex.corpustools.wavaskey import load_wavaskey, load_wavaskeys

###############################################################################
#                                  Constants                                  #
###############################################################################

_SCRIPT_DIR = dirname(realpath(__file__))
CLDB_PATH = join(_SCRIPT_DIR, os.pardir, 'data', 'database.py')

FEAT_SIZE = 4
MIN_CLASSIFIER_COUNT = 4
MIN_FEATURE_COUNT = 3

MODEL_FNAME_TPT = "dailogreg.{utthyp_type}.model"

###############################################################################
#                                  Functions                                  #
###############################################################################


def get_model_fname(parts, utthyp_type):
    model_fname = MODEL_FNAME_TPT.format(utthyp_type=utthyp_type)
    if parts == 'all':
        model_fname += '.all'
    return model_fname


def constructor_for_utthyp(utthyp_type):
    if utthyp_type == 'nbl':
        return UtteranceNBList
    else:
        # What else?
        return Utterance


def increase_weight(d, weight):
    new_d = {}
    for i in range(weight):
        for k in d:
            new_d["{k}v_{i}".format(k=k, i=i)] = d[k]

    d.update(new_d)


def train(fn_model,
          fn_transcriptions, constructor, fn_annotations,
          fn_bs_transcriptions,
          fn_bs_annotations,
          min_feature_count=2,
          min_classifier_count=2,
          limit=100000):
    """
    Trains a SLU DAILogRegClassifier model.

    :param fn_model:
    :param fn_transcriptions:
    :param constructor:
    :param fn_annotations:
    :param fn_bs_transcriptions: Like fn_transcriptions, but for bootstrapped
        data, and only one file, not a list.
    :param fn_bs_annotations: Like fn_annotations, but for bootstrapped data,
        and only one file, not a list.
    :param limit:
    :return:

    """

    # Load the bootstrap utts.
    bs_utterances = load_wavaskey(fn_bs_transcriptions, Utterance,
                                  limit=limit)
    increase_weight(bs_utterances, min_feature_count + 10)
    # Load the bootstrap DAs.
    bs_das = load_wavaskey(fn_bs_annotations, DialogueAct, limit=limit)
    increase_weight(bs_das, min_feature_count + 10)

    # Load usage utts.
    utterances = load_wavaskeys(fn_transcriptions, constructor, limit=limit)
    # Load usage DAs.
    das = load_wavaskeys(fn_annotations, DialogueAct, limit=limit)

    utterances.update(bs_utterances)
    das.update(bs_das)

    cldb = CategoryLabelDatabase(CLDB_PATH)
    preprocessing = PTICSSLUPreprocessing(cldb)
    slu = DAILogRegClassifier(cldb, preprocessing, features_size=FEAT_SIZE)

    slu.extract_classifiers(das, utterances, verbose=True)
    slu.prune_classifiers(min_classifier_count=min_classifier_count)
    slu.print_classifiers()
    slu.gen_classifiers_data()
    slu.prune_features(min_feature_count=min_feature_count, verbose=True)

    slu.train(inverse_regularisation=1e1, verbose=True)

    slu.save_model(fn_model)

###############################################################################
#                           Command-line interface                            #
###############################################################################


def parse_args(argv=None):
    arger = argparse.ArgumentParser(
        description="Trains a MaxEnt SLU classifier for given training data.")
    arger.add_argument('-i', '--input-dirs',
                       metavar='DIR',
                       nargs='+',
                       default=[_SCRIPT_DIR],
                       help='Paths towards directories with training files.')
    arger.add_argument('-b', '--bootstrap-dir',
                       metavar='DIR',
                       default=_SCRIPT_DIR,
                       help='Path towards a directory with bootstrap data.')
    arger.add_argument('-o', '--output-dir',
                       metavar='DIR',
                       default=_SCRIPT_DIR,
                       help='Path towards a directory where trained models '
                            'shall be written. If the directory does not '
                            'exist, it will be created.')

    args = arger.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    indirs = args.input_dirs
    outdir = args.output_dir
    exists(outdir) or os.makedirs(outdir)

    for parts in ('all', 'train'):
        for utthyp_type in ('trn', 'asr', 'nbl'):
            model_fname = get_model_fname(parts, utthyp_type)
            trs_fname = '{parts}.{utthyp_type}'.format(**locals())
            sem_fname = '{parts}.{utthyp_type}.hdc.sem'.format(**locals())

            train(join(outdir, model_fname),
                  [join(indir, trs_fname) for indir in indirs],
                  constructor_for_utthyp(utthyp_type),
                  [join(indir, sem_fname) for indir in indirs],
                  join(args.bootstrap_dir, 'bootstrap.trn'),
                  join(args.bootstrap_dir, 'bootstrap.sem'),
                  min_feature_count=MIN_FEATURE_COUNT,
                  min_classifier_count=MIN_CLASSIFIER_COUNT)


if __name__ == '__main__':
    main()
