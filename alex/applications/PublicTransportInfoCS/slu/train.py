#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import argparse
import codecs
import os
from os.path import exists, join
import sys

if __name__ == "__main__":
    import autopath

from alex.applications.PublicTransportInfoCS.preprocessing import PTICSSLUPreprocessing
from alex.components.asr.utterance import Utterance, UtteranceNBList
from alex.components.slu.da import DialogueAct
from alex.components.slu.base import CategoryLabelDatabase
from alex.components.slu.dailrclassifier import DAILogRegClassifier
from alex.corpustools.wavaskey import load_wavaskey

###############################################################################
#                                  Constants                                  #
###############################################################################

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


def increase_weight(d, weight):
    new_d = {}
    for i in range(weight):
        for k in d:
            new_d["{k}v_{i}".format(k=k, i=i)] = d[k]

    d.update(new_d)


def train(fn_model,
          fn_transcription, constructor, fn_annotation,
          fn_bs_transcription, fn_bs_annotation,
          min_feature_count=2,
          min_classifier_count=2,
          limit=100000):
    """
    Trains a SLU DAILogRegClassifier model.

    :param fn_model:
    :param fn_transcription:
    :param constructor:
    :param fn_annotation:
    :param limit:
    :return:
    """
    bs_utterances = load_wavaskey(fn_bs_transcription, Utterance, limit=limit)
    increase_weight(bs_utterances, min_feature_count + 10)
    bs_das = load_wavaskey(fn_bs_annotation, DialogueAct, limit=limit)
    increase_weight(bs_das, min_feature_count + 10)

    utterances = load_wavaskey(fn_transcription, constructor, limit=limit)
    das = load_wavaskey(fn_annotation, DialogueAct, limit=limit)

    utterances.update(bs_utterances)
    das.update(bs_das)

    cldb = CategoryLabelDatabase('../data/database.py')
    preprocessing = PTICSSLUPreprocessing(cldb)
    slu = DAILogRegClassifier(cldb, preprocessing, features_size=4)

    slu.extract_classifiers(das, utterances, verbose=True)
    slu.prune_classifiers(min_classifier_count=min_classifier_count)
    slu.print_classifiers()
    slu.gen_classifiers_data()
    slu.prune_features(min_feature_count=min_feature_count, verbose=True)

    slu.train(inverse_regularisation=1e1, verbose=True)

    slu.save_model(fn_model)


def parse_args(argv=None):
    arger = argparse.ArgumentParser(
        description="Trains a MaxEnt SLU classifier for given training data.")
    arger.add_argument('-i', '--input-dir',
                       metavar='DIR',
                       help='Path towards a directory with all needed '
                            'training files.')
    arger.add_argument('-o', '--output-dir',
                       metavar='DIR',
                       help='Path towards a directory where trained models '
                            'shall be written. If the directory does not '
                            'exist, it will be created.')

    args = arger.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    indir = args.input_dir
    outdir = args.output_dir
    exists(outdir) or os.makedirs(outdir)

    for parts in ('all', 'train'):
        for utthyp_type in ('trn', 'asr', 'nbl'):
            if utthyp_type in ('trn', 'asr'):
                utthyp_constructor = Utterance
            else:
                utthyp_constructor = UtteranceNBList

            model_fname = get_model_fname(parts, utthyp_type)
            trs_fname = '{parts}.{utthyp_type}'.format(**locals())
            sem_fname = '{parts}.{utthyp_type}.hdc.sem'.format(**locals())

            train(join(outdir, model_fname),
                  join(indir, trs_fname),
                  utthyp_constructor,
                  join(indir, sem_fname),
                  join(indir, 'bootstrap.trn'),
                  join(indir, 'bootstrap.sem'),
                  min_feature_count=MIN_FEATURE_COUNT,
                  min_classifier_count=MIN_CLASSIFIER_COUNT)


if __name__ == '__main__':
    # Make sure we can read and write UTF-8.
    if not sys.stdin.isatty():
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    if not sys.stdout.isatty():
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    if not sys.stderr.isatty():
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

    main()
