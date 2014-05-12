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
from alex.applications.PublicTransportInfoCS.slu.prepare_data import sort_dais
from alex.applications.PublicTransportInfoCS.preprocessing import PTICSSLUPreprocessing
from alex.applications.PublicTransportInfoCS.train import (
    constructor_for_utthyp,
    get_model_fname)
from alex.components.asr.utterance import Utterance, UtteranceNBList
from alex.components.slu.base import CategoryLabelDatabase

###############################################################################
#                                  Constants                                  #
###############################################################################

_SCRIPT_DIR = dirname(realpath(__file__))
CLDB_PATH = join(_SCRIPT_DIR, os.pardir, 'data', 'database.py')
PARTS = ('train', 'dev', 'test')

###############################################################################
#                                  Functions                                  #
###############################################################################


def test_slu(slu, fn_input, constructor, fn_reference, output_fname,
             output_dir=None,
             desc=None):
    """
    Tests a SLU DAILogRegClassifier model.

    :param fn_input:
    :param constructor:
    :param fn_reference:
    :param output_fname:
    :param output_dir: If provided, specifies path towards the directory where
        the output shall be written. If not specified, output is written to the
        same directory as where the input files are read from.
    :param desc: If provided, gives the description of the SLU -- to be logged.
    :return:

    """

    from alex.corpustools.wavaskey import load_wavaskey, save_wavaskey
    from alex.corpustools.semscore import score

    # Interpret arguments.
    if output_dir is None:
        output_dir = dirname(fn_input)

    # Prepare for decoding.
    testing_msg = ('Testing {desc}'.format(desc=desc) if desc is not None
                   else 'Testing SLU')
    print "=" * 60
    print testing_msg
    print "-" * 60

    # Load the data.
    test_utterances = load_wavaskey(fn_input, constructor, limit=100000)

    # Decode the data.
    parsed_das = {}
    for utt_key, utt in sorted(test_utterances.iteritems()):
        if isinstance(utt, Utterance):
            obs = {'utt': utt}
        elif isinstance(utt, UtteranceNBList):
            obs = {'utt_nbl': utt}
        else:
            raise Exception('Unsupported observation type')

        print '-' * 60
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
            print '*' * 60
            print utt
            print dah.da
            slu.parse(obs, verbose=True)

    # Write the parses.
    parses_out_fpath = join(output_dir, output_fname)
    save_wavaskey(parses_out_fpath, parsed_das, trans=sort_dais)

    # Determine where the score should be written.
    scores_fpath = '{outpath}.score'.format(outpath=parses_out_fpath)

    # Write the score.
    with codecs.open(scores_fpath, 'w', encoding='UTF-8') as score_file:
        score(fn_reference, parses_out_fpath, True, True, score_file)


def trained_slu_test(fn_model, fn_input, constructor, fn_reference,
                     output_dir=None):
    """
    Tests a SLU DAILogRegClassifier model.

    :param fn_model:
    :param fn_input:
    :param constructor:
    :param fn_reference:
    :param output_dir: If provided, specifies path towards the directory where
        the output shall be written. If not specified, output is written to the
        same directory as where the input files are read from.
    :return:

    """

    # Construct the SLU.
    from alex.components.slu.dailrclassifier import DAILogRegClassifier

    cldb = CategoryLabelDatabase(CLDB_PATH)
    preprocessing = PTICSSLUPreprocessing(cldb)
    slu = DAILogRegClassifier(cldb, preprocessing)

    slu.load_model(fn_model)

    # Construct the SLU description.
    desc = 'log-reg SLU: {}'.format(' '.join((fn_model,
                                              fn_input,
                                              fn_reference)))

    # Determine the output filename.
    if 'trn' in fn_model:
        extension = 'model.trn.sem.out'
    elif 'asr' in fn_model:
        extension = 'model.asr.sem.out'
    elif 'nbl' in fn_model:
        extension = 'model.nbl.sem.out'
    else:
        extension = 'XXX.sem.out'
    output_fname = '{bn}.{ext}'.format(bn=basename(fn_input),
                                       ext=extension)

    test_slu(slu, fn_input, constructor, fn_reference, output_fname,
             output_dir, desc)


def hdc_slu_test(fn_input, constructor, fn_reference, output_dir=None):
    """
    Tests a SLU DAILogRegClassifier model.

    :param fn_input:
    :param constructor:
    :param fn_reference:
    :param output_dir: If provided, specifies path towards the directory where
        the output shall be written. If not specified, output is written to the
        same directory as where the input files are read from.
    :return:

    """

    # Construct the SLU.
    from alex.applications.PublicTransportInfoCS.hdc_slu import PTICSHDCSLU

    cldb = CategoryLabelDatabase(CLDB_PATH)
    preprocessing = PTICSSLUPreprocessing(cldb)
    slu = PTICSHDCSLU(preprocessing)

    # Construct the SLU description.
    desc = 'HDC SLU: {}'.format(' '.join((fn_input, fn_reference)))

    # Determine the output filename.
    output_fname = '{base}.hdc.slu.sem.out'.format(base=basename(fn_input))

    test_slu(slu, fn_input, constructor, fn_reference, output_fname,
             output_dir, desc)


def eval_all_on_all(models_dir, data_dir, output_dir=None):
    all_refs_fpath = join(data_dir, 'all.trn.hdc.sem')
    for utthyp_type in ('trn', 'asr', 'nbl'):
        all_utts_fpath = join(data_dir, 'all.{typ}'.format(typ=utthyp_type))
        utthyp_constructor = constructor_for_utthyp(utthyp_type)

        hdc_slu_test(all_utts_fpath, utthyp_constructor, all_refs_fpath,
                     output_dir)

        model_fname = get_model_fname('all', utthyp_type)
        model_fpath = join(models_dir, model_fname)
        trained_slu_test(model_fpath,
                         all_utts_fpath,
                         utthyp_constructor,
                         all_refs_fpath,
                         output_dir)


def eval_crossparts(models_dir, data_dir, output_dir=None):
    for part in PARTS:
        refs_fname = '{part}.trn.hdc.sem'.format(part=part)
        refs_fpath = join(data_dir, refs_fname)

        for utthyp_type in ('trn', 'asr', 'nbl'):
            model_fname = get_model_fname(part, utthyp_type)
            model_fpath = join(models_dir, model_fname)

            utts_fname = '{part}.{utthyp_type}'.format(**locals())
            utts_fpath = join(data_dir, utts_fname)

            utthyp_constructor = constructor_for_utthyp(utthyp_type)

            trained_slu_test(model_fpath,
                             utts_fpath,
                             utthyp_constructor,
                             refs_fpath,
                             output_dir)


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


def main(argv=None):
    args = parse_args(argv)

    # CHEATING: experiment on all data using models trained on all data
    for data_dir in args.input_dirs:
        eval_all_on_all(args.models_dir, data_dir, args.output_dir)

    # REGULAR EXPERIMENT: evaluating models trained on training data and evaluated on deb and test data
    # **WARNING** due to data sparsity the metrics on the dev and test data fluctuate a lot
    # therefore meaning full results can be only obtained using N-fold cross validation
    for data_dir in args.input_dirs:
        eval_crossparts(args.models_dir, data_dir, args.output_dir)


if __name__ == "__main__":
    main()


# pymode:lint_ignore=E501,E221:
