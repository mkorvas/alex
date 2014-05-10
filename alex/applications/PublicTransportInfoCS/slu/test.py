#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import argparse
import codecs
import os
from os.path import basename, dirname, join, realpath

if __name__ == "__main__":
    import autopath

# XXX This function should really be defined somewhere else than prepare_data.
from alex.applications.PublicTransportInfoCS.prepare_data import sort_dais
from alex.applications.PublicTransportInfoCS.train import (
    constructor_for_utthyp,
    get_model_fname)
from alex.components.asr.utterance import Utterance, UtteranceNBList

###############################################################################
#                                  Constants                                  #
###############################################################################

_SCRIPT_DIR = dirname(realpath(__file__))
CLDB_PATH = join(_SCRIPT_DIR, os.pardir, 'data', 'database.py')
PARTS = ('train', 'dev', 'test')

###############################################################################
#                                  Functions                                  #
###############################################################################


def trained_slu_test(fn_model, fn_input, constructor, fn_reference):
    """
    Tests a SLU DAILogRegClassifier model.

    :param fn_model:
    :param fn_input:
    :param constructor:
    :param fn_reference:
    :return:
    """
    print "=" * 120
    print "Testing: ", fn_model, fn_input, fn_reference
    print "-" * 120

    from alex.applications.PublicTransportInfoCS.preprocessing import PTICSSLUPreprocessing
    from alex.components.slu.base import CategoryLabelDatabase
    from alex.components.slu.dailrclassifier import DAILogRegClassifier
    from alex.corpustools.wavaskey import load_wavaskey, save_wavaskey
    from alex.corpustools.semscore import score

    cldb = CategoryLabelDatabase(CLDB_PATH)
    preprocessing = PTICSSLUPreprocessing(cldb)
    slu = DAILogRegClassifier(cldb, preprocessing)

    slu.load_model(fn_model)

    test_utterances = load_wavaskey(fn_input, constructor, limit=100000)

    parsed_das = {}
    for utt_key, utt in sorted(test_utterances.iteritems()):
        if isinstance(utt, Utterance):
            obs = {'utt': utt}
        elif isinstance(utt, UtteranceNBList):
            obs = {'utt_nbl': utt}
        else:
            raise BaseException('Unsupported observation type')

        print '-' * 120
        print "Observation:"
        print utt_key, " ==> "
        print unicode(utt)

        da_confnet = slu.parse(obs, verbose=False)

        print "Conf net:"
        print unicode(da_confnet)

        da_confnet.prune()
        dah = da_confnet.get_best_da_hyp()

        print "1 best: "
        print unicode(dah)

        parsed_das[utt_key] = dah.da

        if 'CL_' in str(dah.da):
            print '*' * 120
            print utt
            print dah.da
            slu.parse(obs, verbose=True)

    if 'trn' in fn_model:
        fn_sem = basename(fn_input) + '.model.trn.sem.out'
    elif 'asr' in fn_model:
        fn_sem = basename(fn_input) + '.model.asr.sem.out'
    elif 'nbl' in fn_model:
        fn_sem = basename(fn_input) + '.model.nbl.sem.out'
    else:
        fn_sem = basename(fn_input) + '.XXX.sem.out'

    save_wavaskey(fn_sem, parsed_das, trans=sort_dais)

    f = codecs.open(basename(fn_sem) + '.score', 'w+', encoding='UTF-8')
    score(fn_reference, fn_sem, True, True, f)
    f.close()


def hdc_slu_test(fn_input, constructor, fn_reference):
    """
    Tests a SLU DAILogRegClassifier model.

    :param fn_model:
    :param fn_input:
    :param constructor:
    :param fn_reference:
    :return:
    """
    print "=" * 120
    print "Testing HDC SLU: ", fn_input, fn_reference
    print "-" * 120

    from alex.components.slu.base import CategoryLabelDatabase
    from alex.applications.PublicTransportInfoCS.preprocessing import PTICSSLUPreprocessing
    from alex.applications.PublicTransportInfoCS.hdc_slu import PTICSHDCSLU
    from alex.corpustools.wavaskey import load_wavaskey, save_wavaskey
    from alex.corpustools.semscore import score

    cldb = CategoryLabelDatabase(CLDB_PATH)
    preprocessing = PTICSSLUPreprocessing(cldb)
    hdc_slu = PTICSHDCSLU(preprocessing)

    test_utterances = load_wavaskey(fn_input, constructor, limit=100000)

    parsed_das = {}
    for utt_key, utt in sorted(test_utterances.iteritems()):
        if isinstance(utt, Utterance):
            obs = {'utt': utt}
        elif isinstance(utt, UtteranceNBList):
            obs = {'utt_nbl': utt}
        else:
            raise BaseException('Unsupported observation type')

        print '-' * 120
        print "Observation:"
        print utt_key, " ==> "
        print unicode(utt)

        da_confnet = hdc_slu.parse(obs, verbose=False)

        print "Conf net:"
        print unicode(da_confnet)

        da_confnet.prune()
        dah = da_confnet.get_best_da_hyp()

        print "1 best: "
        print unicode(dah)

        parsed_das[utt_key] = dah.da

        if 'CL_' in str(dah.da):
            print '*' * 120
            print utt
            print dah.da
            hdc_slu.parse(obs, verbose=True)

    fn_sem = basename(fn_input) + '.hdc.slu.sem.out'

    save_wavaskey(fn_sem, parsed_das, trans=sort_dais)

    f = codecs.open(basename(fn_sem) + '.score', 'w+', encoding='UTF-8')
    score(fn_reference, fn_sem, True, True, f)
    f.close()


def parse_args(argv=None):
    arger = argparse.ArgumentParser(
        description="Trains a MaxEnt SLU classifier for given training data.")
    arger.add_argument('-M', '--models-dir',
                       metavar='DIR',
                       default=_SCRIPT_DIR,
                       help='Path towards a directory with trained SLU models.'
                       )
    arger.add_argument('-i', '--input-dirs',
                       metavar='DIR',
                       nargs='+',
                       default=[_SCRIPT_DIR],
                       help='Paths towards directories with prepared SLU '
                            'utts/SLU-hyps files.')
    arger.add_argument('-o', '--output-dir',
                       metavar='DIR',
                       default=_SCRIPT_DIR,
                       help='Path towards a directory where trained models '
                            'shall be written. If the directory does not '
                            'exist, it will be created.')

    args = arger.parse_args(argv)
    return args


def eval_all_on_all(models_dir, data_dir):
    all_refs_fpath = join(data_dir, 'all.trn.hdc.sem')
    for utthyp_type in ('trn', 'asr', 'nbl'):
        all_utts_fpath = join(data_dir, 'all.{typ}'.format(typ=utthyp_type))
        utthyp_constructor = constructor_for_utthyp(utthyp_type)

        hdc_slu_test(all_utts_fpath, utthyp_constructor, all_refs_fpath)

        model_fname = get_model_fname('all', utthyp_type)
        model_fpath = join(models_dir, model_fname)
        trained_slu_test(model_fpath,
                         all_utts_fpath,
                         utthyp_constructor,
                         all_refs_fpath)


def eval_crossparts(models_dir, data_dir):
    for part in PARTS:
        refs_fname = '{part}.trn.hdc.sem'.format(part=part)
        refs_fpath = join(data_dir, refs_fname)

        for utthyp_type in ('trn', 'asr', 'nbl'):
            model_fname = get_model_fname(part, utthyp_type)
            model_fpath = join(models_dir, model_fname)

            utts_fname = '{part}.{utthyp_type}'.format(**locals())
            utts_fpath = join(data_dir, utts_fname)

            utthyp_constructor = constructor_for_utthyp(utthyp_type)

            trained_slu_test(model_fpath, utts_fpath, utthyp_constructor,
                             refs_fpath)


def main(argv=None):
    args = parse_args(argv)

    # CHEATING: experiment on all data using models trained on all data
    for data_dir in args.input_dirs:
        eval_all_on_all(args.models_dir, data_dir)

    # REGULAR EXPERIMENT: evaluating models trained on training data and evaluated on deb and test data
    # **WARNING** due to data sparsity the metrics on the dev and test data fluctuate a lot
    # therefore meaning full results can be only obtained using N-fold cross validation
    for data_dir in args.input_dirs:
        eval_crossparts(args.models_dir, data_dir)


if __name__ == "__main__":
    main()
