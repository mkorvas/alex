#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" The script has two commands:

--fast      it approximates SLU output on N-best lists by SLU output from 1-best
--uniq      it generates only files with unique texts and their SLU HDC output
--asr-log   it uses the asr hypotheses from call logs

"""

from __future__ import unicode_literals

import argparse
from collections import namedtuple
import os
from os.path import dirname, exists, join, realpath
import random
import sys
import xml.dom.minidom

# Add Alex on Pythonpath.
if __name__ == "__main__":
    import autopath

from alex.utils.various import get_text_from_xml_node

from alex.applications.PublicTransportInfoCS.hdc_slu import PTICSHDCSLU
from alex.applications.PublicTransportInfoCS.preprocessing import PTICSSLUPreprocessing
from alex.components.asr.common import asr_factory
from alex.components.asr.utterance import Utterance, UtteranceNBList
from alex.components.slu.base import CategoryLabelDatabase
from alex.corpustools.text_norm_cs import normalise_text, exclude_slu
from alex.corpustools.wavaskey import save_wavaskey
from alex.utils.config import Config
from alex.utils.fs import find

###############################################################################
#                                   Classes                                   #
###############################################################################


class InvalidTurnException(Exception):
    pass

###############################################################################
#                                  Constants                                  #
###############################################################################

_SCRIPT_DIR = dirname(realpath(__file__))
PARTS = ('train', 'dev', 'test')
Partitioned = namedtuple('Partitioned', PARTS)

SEED_VALUE     = 10

TRAIN_TRIP_REL = 0.8  # trip for the train partition, relative
DEV_TRIP_REL   = 0.9  # trip for the dev partition, relative
                      # ...the rest goes to test

CLDB_PATH = join(_SCRIPT_DIR, os.pardir, 'data', 'database.py')
KALDI_CFG_PATH = join(_SCRIPT_DIR, os.pardir, 'kaldi.cfg')

INDOMAIN_DATA        = join(_SCRIPT_DIR, 'indomain_data')
TRANSCRIBED_FNAME    = 'asr_transcribed.xml'

FN_UNIQ_TRN          = 'uniq.trn'
FN_UNIQ_TRN_HDC_SEM  = 'uniq.trn.hdc.sem'
FN_UNIQ_TRN_SEM      = 'uniq.trn.sem'

FN_ALL_SEM           = 'all.sem'
FN_ALL_TRN           = 'all.trn'
FN_ALL_TRN_HDC_SEM   = 'all.trn.hdc.sem'
FN_ALL_ASR           = 'all.asr'
FN_ALL_ASR_HDC_SEM   = 'all.asr.hdc.sem'
FN_ALL_NBL           = 'all.nbl'
FN_ALL_NBL_HDC_SEM   = 'all.nbl.hdc.sem'

FN_TPT_TRN           = '{part}.trn'
FN_TPT_HDC_SEM       = '{part}.trn.hdc.sem'
FN_TPT_ASR           = '{part}.asr'
FN_TPT_ASR_HDC_SEM   = '{part}.asr.hdc.sem'
FN_TPT_NBL           = '{part}.nbl'
FN_TPT_NBL_HDC_SEM   = '{part}.nbl.hdc.sem'

###############################################################################
#                             Auxiliary functions                             #
###############################################################################


def controlled_shuffle(set_):
    random.seed(SEED_VALUE)
    random.shuffle(set_)


def split_to_parts(set_):
    train_trip = int(TRAIN_TRIP_REL * len(set_))
    dev_trip = int(DEV_TRIP_REL * len(set_))

    train = set_[:train_trip]
    dev = set_[train_trip:dev_trip]
    test = set_[dev_trip:]

    return Partitioned(train, dev, test)


def save_partitions(parted, fname_tpt, trans=None):
    """Saves partitioned data in the wav-as-key format to a file.

    :param parted: tuple of collections of data to be saved, one collection for
                   each partition
    :rtype: Partitioned

    :param fname_tpt: template string for the filename where to write
    :rtype: basestring

    """

    for part in PARTS:
        save_wavaskey(fname_tpt.format(part=part),
                      dict(getattr(parted, part)),
                      trans=trans)


def split_and_save(set_, fname_tpt, trans=None):
    save_partitions(split_to_parts(set_), fname_tpt, trans)

###############################################################################
#                           Main business functions                           #
###############################################################################


def sort_dais(da):
    return '&'.join(sorted(unicode(da).split('&')))


def format_tup_for_wavaskey(tup):
    return " <=> ".join(map(unicode, tup))


def normalise_semi_words(txt):
    # normalise these semi-words
    if txt == '__other__':
        txt = '_other_'
    elif txt == '__silence__':
        txt = '_other_'
    elif not txt:
        txt = '_other_'

    return txt


MSG_SKIPPING_TURN_ASRS = "Skipping turn {turn} in file {fn} - asrs: {asrs}"
MSG_SKIPPING_TURN_RECS = "Skipping turn {turn} in file {fn} - recs: {recs}"
MSG_SKIPPING_TURN_TRANS = "Skipping turn {turn} in file {fn} - trans: {trans}"
MSG_SKIPPING_TURN_NEXT_ASRS = ("Skipping turn {turn} in file: {fn} - "
                               "asrs: {asrs} - next_asrs: {next_asrs}")
MSG_DELAYED_ASR = ("Recovered from missing ASR output by using a delayed ASR "
                   "output from the following turn of turn {turn}. File: {fn} "
                   "- next_asrs: {asrs}")
MSG_EXTRA_ASR = ("Recovered from EXTRA ASR outputs by using the last ASR "
                 "output from the turn. File: {fn} - asrs: {asrs}")


def get_trans_for_turn(turns, turn_idx, fname='<filename>'):
    turn = turns[turn_idx]
    trans = turn.getElementsByTagName("asr_transcription")

    if len(trans) == 0:
        print >>sys.stderr, MSG_SKIPPING_TURN_TRANS.format(fn=fname,
                                                           trans=len(trans))
        raise InvalidTurnException()

    return trans


def get_recs_for_turn(turns, turn_idx, fname='<filename>'):
    turn = turns[turn_idx]
    recs = turn.getElementsByTagName("rec")

    if len(recs) != 1:
        print >>sys.stderr, MSG_SKIPPING_TURN_RECS.format(turn=turn_idx,
                                                          fn=fname,
                                                          recs=len(recs))
        raise InvalidTurnException()

    return recs


def get_hyps_for_turn(turns, turn_idx, fname='<filename>'):
    turn = turns[turn_idx]
    asrs = turn.getElementsByTagName("asr")

    if len(asrs) == 0 and (turn_idx + 1) < len(turns):
        next_asrs = turns[turn_idx + 1].getElementsByTagName("asr")
        if len(next_asrs) != 2:
            print >>sys.stderr, MSG_SKIPPING_TURN_NEXT_ASRS.format(turn=turn_idx, fn=fname, asrs=len(asrs), next_asrs=len(next_asrs))
            raise InvalidTurnException()

        print >>sys.stderr, MSG_DELAYED_ASR.format(turn=turn_idx, fn=fname, asrs=len(next_asrs))
        return next_asrs[0].getElementsByTagName("hypothesis")

    elif len(asrs) == 1:
        return asrs[0].getElementsByTagName("hypothesis")

    elif len(asrs) == 2:
        print >>sys.stderr, MSG_EXTRA_ASR.format(fn=fname, asrs=len(asrs))
        return asrs[-1].getElementsByTagName("hypothesis")

    else:
        print >>sys.stderr, MSG_SKIPPING_TURN_ASRS.format(turn=turn_idx, fn=fname, asrs=len(asrs))
        raise InvalidTurnException()


def slu_wavaskey_parse(wav_key, text, slu):
    return (wav_key, slu.parse_1_best({'utt': Utterance(text)}).get_best_da())


def read_trns_and_sems(infiles, asr_log=False, uniq=False, fast=False,
                       limit=100000):

    cldb = CategoryLabelDatabase(CLDB_PATH)
    preprocessing = PTICSSLUPreprocessing(cldb)
    slu = PTICSHDCSLU(preprocessing)
    cfg = Config.load_configs([KALDI_CFG_PATH], use_default=True)
    asr_rec = asr_factory(cfg)

    sem = []
    trn = []
    trn_hdc_sem = []
    asr = []
    asr_hdc_sem = []
    nbl = []
    nbl_hdc_sem = []

    for fn in infiles[:limit]:
        f_dir = dirname(fn)

        print >>sys.stderr, "Processing:", fn
        doc = xml.dom.minidom.parse(fn)
        turns = doc.getElementsByTagName("turn")

        for turn_idx, turn in enumerate(turns):
            if turn.getAttribute('speaker') != 'user':
                continue

            try:
                trans = get_trans_for_turn(turns, turn_idx, fn)
                recs = get_recs_for_turn(turns, turn_idx, fn)
                hyps = get_hyps_for_turn(turns, turn_idx, fn)
            except InvalidTurnException:
                continue

            wav_key = recs[0].getAttribute('fname')
            wav_path = join(f_dir, wav_key)

            # FIXME: Check whether the last transcription is really the best! FJ
            transcription = get_text_from_xml_node(trans[-1])
            transcription = normalise_text(transcription)

            if not asr_log:
                asr_rec_nbl = asr_rec.rec_wav_file(wav_path)
                asr_1best = unicode(asr_rec_nbl.get_best())
            else:
                asr_1best = get_text_from_xml_node(hyps[0])
                asr_1best = normalise_semi_words(asr_1best)

            if exclude_slu(transcription) or 'DOM Element:' in asr_1best:
                print >>sys.stderr, "Skipping transcription:", unicode(transcription)
                print >>sys.stderr, "Skipping ASR output:   ", unicode(asr_1best)
                continue

            # The silence does not have asr_1best label in the language model.
            transcription = transcription.replace('_SIL_', '')

            trn.append((wav_key, transcription))

            print >>sys.stderr, "Parsing transcription:", unicode(transcription)
            print >>sys.stderr, "                  ASR:", unicode(asr_1best)

            # HDC SLU on transcription
            trn_hdc_sem.append(slu_wavaskey_parse(wav_key, transcription, slu))

            if not uniq:
                # HDC SLU on 1 best ASR
                asr.append((wav_key, asr_1best))
                _, slu_1best_asr = slu_wavaskey_parse(wav_key, asr_1best, slu)
                asr_hdc_sem.append((wav_key, slu_1best_asr))

                # HDC SLU on N best ASR
                nblist = UtteranceNBList()
                if not asr_log:
                    nblist = asr_rec_nbl

                    print >>sys.stderr, 'ASR RECOGNITION NBLIST\n', unicode(nblist)
                else:
                    for hyp in hyps:
                        txt = get_text_from_xml_node(hyp)
                        txt = normalise_semi_words(txt)

                        nblist.add(abs(float(hyp.getAttribute('p'))), Utterance(txt))

                nblist.merge()
                nblist.normalise()

                nbl.append((wav_key, nblist.serialise()))

                if fast:
                    nbl_hdc_sem.append((wav_key, slu_1best_asr))
                else:
                    slu_1best_nbl = slu.parse_nblist({'utt_nbl': nblist}).get_best_da()
                    nbl_hdc_sem.append((wav_key, slu_1best_nbl))

            # there is no manual semantics in the transcriptions yet
            sem.append((wav_key, None))

    return ((trn, trn_hdc_sem), (asr, asr_hdc_sem), (nbl, nbl_hdc_sem)), sem


def parse_args(argv=None):
    arger = argparse.ArgumentParser(
        description="Transforms data from XML files with transcribed and "
                    "annotated dialogues in a CUED-compliant format into "
                    "the format needed for training Alex SLU models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arger.add_argument('-i', '--input-dir',
                       default=INDOMAIN_DATA,
                       help='Path towards the directory below which annotated '
                            'XML files can be found.')
    arger.add_argument('-o', '--output-dir',
                       default=_SCRIPT_DIR,
                       help='Path towards the directory into which output '
                            'files shall be written. If it does not exist, '
                            'it will be created.')
    arger.add_argument('--asr-log',
                       action='store_true',
                       help='Use ASR hyps from logs rather than decoding them '
                            'using a current ASR system.')
    arger.add_argument('--fast',
                       action='store_true',
                       help='Do not build SLU hyps from whole n-best lists, '
                            'only from their first best ASR hypothesis.')
    arger.add_argument('--uniq',
                       action='store_true',
                       help='Do not produce data based on ASR hypotheses.')

    args = arger.parse_args(argv)

    return args


def main():
    args = parse_args()
    outdir = args.output_dir
    if not exists(outdir):
        os.makedirs(outdir)

    print >>sys.stderr, "Generating the SLU train and test data"
    print >>sys.stderr, "-" * 60

    ###########################################################################

    transcription_fnames = sorted(find(args.input_dir, TRANSCRIBED_FNAME,
                                       mindepth=1, maxdepth=6))

    ((trn, trn_hdc_sem), (asr, asr_hdc_sem), (nbl, nbl_hdc_sem)), sem = (
        read_trns_and_sems(transcription_fnames,
                           args.asr_log, args.uniq, args.fast))

    # Construct corresponding uniqued dicts.
    uniq_trn = {}
    uniq_trn_hdc_sem = {}
    uniq_trn_sem = {}
    trn_set = set()
    sem = dict(trn_hdc_sem)

    for wav_key, transcription in trn:
        if transcription not in trn_set:
            trn_set.add(transcription)
            uniq_trn[wav_key] = transcription
            uniq_trn_hdc_sem[wav_key] = sem[wav_key]
            uniq_trn_sem[wav_key] = (transcription, unicode(sem[wav_key]))

    save_wavaskey(join(outdir, FN_UNIQ_TRN), uniq_trn)
    save_wavaskey(join(outdir, FN_UNIQ_TRN_HDC_SEM),
                  uniq_trn_hdc_sem,
                  trans=sort_dais)
    save_wavaskey(join(outdir, FN_UNIQ_TRN_SEM),
                  uniq_trn_sem,
                  trans=format_tup_for_wavaskey)

    # all
    save_wavaskey(join(outdir, FN_ALL_TRN), dict(trn))
    save_wavaskey(join(outdir, FN_ALL_TRN_HDC_SEM),
                  dict(trn_hdc_sem),
                  trans=sort_dais)

    if not args.uniq:
        save_wavaskey(join(outdir, FN_ALL_ASR), dict(asr))
        save_wavaskey(join(outdir, FN_ALL_ASR_HDC_SEM),
                      dict(asr_hdc_sem),
                      trans=sort_dais)

        save_wavaskey(join(outdir, FN_ALL_NBL), dict(nbl))
        save_wavaskey(join(outdir, FN_ALL_NBL_HDC_SEM),
                      dict(nbl_hdc_sem),
                      trans=sort_dais)

        map(controlled_shuffle,
            (trn, trn_hdc_sem, asr, asr_hdc_sem, nbl, nbl_hdc_sem))

        split_and_save(trn, join(outdir, FN_TPT_TRN))
        split_and_save(trn_hdc_sem, join(outdir, FN_TPT_HDC_SEM), sort_dais)

        split_and_save(asr, join(outdir, FN_TPT_ASR))
        split_and_save(asr_hdc_sem,
                       join(outdir, FN_TPT_ASR_HDC_SEM),
                       sort_dais)

        split_and_save(nbl, join(outdir, FN_TPT_NBL))
        split_and_save(nbl_hdc_sem,
                       join(outdir, FN_TPT_NBL_HDC_SEM),
                       sort_dais)

if __name__ == '__main__':
    main()

# pymode:lint_ignore=E501,E221:
