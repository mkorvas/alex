#!/usr/bin/env python
# encoding: utf-8
from __future__ import unicode_literals

"""
Extracts wavs from call logs
and runs the Kaldi decoding using the AM and HCLG graph from models directory
"""

import os
import errno
import xml.dom.minidom
import fnmatch
import argparse
import time

import autopath

import alex.utils.various as various

from alex.components.asr.common import asr_factory
from alex.components.asr.utterance import Utterance
from alex.components.hub.messages import Frame
from alex.corpustools.text_norm_cs import normalise_text, exclude_lm
from alex.corpustools.wavaskey import save_wavaskey, load_wavaskey
from alex.corpustools.asrscore import score
from alex.utils.config import Config
from alex.utils.audio import load_wav, wav_duration


def rec_wav_file(asr, cfg, wav_path):
    pcm = load_wav(cfg, wav_path)
    frame = Frame(pcm)

    start = time.time()
    asr.rec_in(frame)
    rec_in_end = time.time()
    res = asr.hyp_out()
    hyp_out_end = time.time()

    asr.flush()

    return res, rec_in_end - start, hyp_out_end - rec_in_end


def decode_info(asr, cfg, wav_path, reference=None):
    print "-"*120
    print
    print '    Wav file:  %s' % str(wav_path)
    print
    print '    Reference: %s' % reference

    wav_dur = wav_duration(wav_path)

    dec_trans, rec_in_dur, hyp_out_dur = rec_wav_file(asr, cfg, wav_path)
    fw_dur = rec_in_dur
    dec_dur = max(rec_in_dur, wav_dur) + hyp_out_dur
    best = unicode(dec_trans.get_best())

    print '    Decoded:   %s' % best
    print '    Wav dur:   %.2f' % wav_dur
    print '    Dec dur:   %.2f' % dec_dur
    print '    FW dur:    %.2f' % fw_dur

    print
    print '    NBest list:'
    print '    ' + u'\n    '.join(['%.5f %s' % (p, t) for p, t in dec_trans.n_best if p > 0.0001])

    return best, dec_dur, fw_dur, wav_dur


def compute_rt_factor(outdir, trn_dict, dec_dict, wavlen_dict, declen_dict, fwlen_dict):
    reference = os.path.join(outdir, 'ref_trn.txt')
    hypothesis = os.path.join(outdir, 'dec_trn.txt')
    save_wavaskey(reference, trn_dict)
    save_wavaskey(hypothesis, dec_dict)
    save_wavaskey(os.path.join(outdir, 'wavlen.txt'), wavlen_dict)
    save_wavaskey(os.path.join(outdir, 'dec_duration.txt'), declen_dict)

    rtf, latency, fw_rtf, fw_latency, d_tot, w_tot = [], [], [], [], 0, 0
    for k in declen_dict.keys():
        w, d, f = wavlen_dict[k], declen_dict[k], fwlen_dict[k]
        d_tot, w_tot = d_tot + d, w_tot + w
        rtf.append(float(d) / w)
        latency.append(float(d) - w)
        fw_rtf.append(float(f) / w)
        fw_latency.append(max(float(f) - w, 0))
    rtf_global = float(d_tot) / w_tot

    print
    print """    # waws:                  %d""" % len(rtf)
    print """    Global RTF mean:         %(rtfglob)f""" % {'rtfglob': rtf_global}

    try:
        rtf.sort()
        rm = rtf[int(len(rtf)*0.5)]
        rc95 = rtf[int(len(rtf)*0.95)]

        print """    RTF median:              %(median)f  RTF       < %(c95)f [in 95%%]""" % {'median': rm, 'c95': rc95}
    except:
        pass

    try:
        fw_rtf.sort()
        fm = fw_rtf[int(len(fw_rtf)*0.5)]
        fc95 = fw_rtf[int(len(fw_rtf)*0.95)]

        print """    Forward RTF median:      %(median)f  FWRTF     < %(c95)f [in 95%%]""" % {'median': fm, 'c95': fc95}
    except:
        pass

    try:
        latency.sort()
        lm = latency[int(len(latency)*0.5)]
        lc95 = latency[int(len(latency)*0.95)]

        print """    Latency median:          %(median)f  Latency   < %(c95)f [in 95%%]""" % {'median': lm, 'c95': lc95}
    except:
        pass

    try:
        fw_latency.sort()
        flm = fw_latency[int(len(fw_latency)*0.5)]
        flc95 = fw_latency[int(len(fw_latency)*0.95)]

        print """    Forward latency median:  %(median)f  FWLatency < %(c95)f [in 95%%]
    """ % {'median': flm, 'c95': flc95}
    except:
        pass

    try:
        print "    95%RTF={rtf:0.2f} 95%FWRTF={fwrtf:0.2f} " \
              "95%LAT={lat:0.2f} 95%FWLAT={fwlat:0.2f}".format(rtf=rc95, fwrtf=fc95, lat=lc95, fwlat=flc95)
        print
    except:
        pass

def compute_save_stat(outdir, trn_dict, dec_dict, wavlen_dict, declen_dict, fwlen_dict):

    compute_rt_factor(outdir, trn_dict, dec_dict, wavlen_dict, declen_dict, fwlen_dict)

    reference = os.path.join(outdir, 'ref_trn.txt')
    hypothesis = os.path.join(outdir, 'dec_trn.txt')

    score(reference, hypothesis)
    # corr, sub, dels, ins, wer, nwords = score_file(trn_dict, dec_dict)

    # print """
    # Please note that the scoring is implicitly ignoring all non-speech events.

    # |==============================================================================================|
    # |            | # Sentences  |  # Words  |   Corr   |   Sub    |   Del    |   Ins    |   Err    |
    # |----------------------------------------------------------------------------------------------|
    # | Sum/Avg    |{num_sents:^14}|{num_words:^11.0f}|{corr:^10.2f}|{sub:^10.2f}|{dels:^10.2f}|{ins:^10.2f}|{wer:^10.2f}|
    # |==============================================================================================|
    # """.format(num_sents=len(trn_dict), num_words=nwords, corr=corr, sub=sub, dels=dels, ins=ins, wer=wer)


def decode_with_reference(reference, outdir, cfg):
    asr = asr_factory(cfg)
    trn_dict = load_wavaskey(reference, Utterance)
    declen_dict, fwlen_dict, wavlen_dict, dec_dict = {}, {}, {}, {}

    for wav_path, reference in trn_dict.iteritems():
        best, dec_dur, fw_dur, wav_dur = decode_info(asr, cfg, wav_path, reference)
        dec_dict[wav_path] = best
        wavlen_dict[wav_path] = wav_dur
        declen_dict[wav_path] = dec_dur
        fwlen_dict[wav_path] = fw_dur

        compute_rt_factor(outdir, trn_dict, dec_dict, wavlen_dict, declen_dict, fwlen_dict)

    compute_save_stat(outdir, trn_dict, dec_dict, wavlen_dict, declen_dict, fwlen_dict)


def extract_from_xml(indomain_data_dir, outdir, cfg):
    glob = 'asr_transcribed.xml'
    asr = asr_factory(cfg)

    print 'Collecting files under %s with glob %s' % (indomain_data_dir, glob)
    files = []
    for root, dirnames, filenames in os.walk(indomain_data_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, glob):
            files.append(os.path.join(root, filename))

    # DEBUG example
    # files = [
    #     '/ha/projects/vystadial/data/call-logs/2013-05-30-alex-aotb-prototype/part1/2013-06-27-09-33-25.116055-CEST-00420221914256/asr_transcribed.xml']

    try:
        trn, dec, dec_len, wav_len = [], [], [], []
        for fn in files:
            doc = xml.dom.minidom.parse(fn)
            turns = doc.getElementsByTagName("turn")
            f_dir = os.path.dirname(fn)

            for turn in turns:
                if turn.getAttribute('speaker') != 'user':
                    continue

                recs = turn.getElementsByTagName("rec")
                trans = turn.getElementsByTagName("asr_transcription")

                if len(recs) != 1:
                    print "Skipping a turn {turn} in file: {fn} - recs: {recs}".format(turn=turn.getAttribute('turn_number'), fn=fn, recs=len(recs))
                    continue

                if len(trans) == 0:
                    print "Skipping a turn in {fn} - trans: {trans}".format(fn=fn, trans=len(trans))
                    continue

                wav_file = recs[0].getAttribute('fname')
                # FIXME: Check whether the last transcription is really the best! FJ
                t = various.get_text_from_xml_node(trans[-1])
                t = normalise_text(t)

                if exclude_lm(t):
                    continue

                # TODO is it still valid? OP
                # The silence does not have a label in the language model.
                t = t.replace('_SIL_', '')
                trn.append((wav_file, t))

                wav_path = os.path.join(f_dir, wav_file)
                best, dec_dur, fw_dur, wav_dur = decode_info(asr, cfg, wav_path, t)
                dec.append((wav_file, best))
                wav_len.append((wav_file, wav_dur))
                dec_len.append((wav_file, dec_dur))

    except Exception as e:
        print 'PARTIAL RESULTS were saved to %s' % outdir
        print e
        raise e
    finally:
        trn_dict = dict(trn)
        dec_dict = dict(dec)
        wavlen_dict = dict(wav_len)
        declen_dict = dict(dec_len)
        compute_save_stat(outdir, trn_dict, dec_dict, wavlen_dict, declen_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=""" TODO """)

    parser.add_argument('-c', '--configs', nargs='+', help='additional configuration files')
    parser.add_argument('-o', '--out-dir', default='decoded',
                        help='The computed statistics are saved the out-dir directory')
    parser.add_argument('-f', default=False, action='store_true',
                        help='If out-dir exists write the results there anyway')

    subparsers = parser.add_subparsers(dest='command',
                                       help='Either extract wav list from xml or expect reference and wavs')

    parser_a = subparsers.add_parser(
        'extract', help='extract wav from all asr_transcribed.xml in directory')
    parser_a.add_argument('indomain_data_dir',
                          help='Directory which should contain symlinks or directories with transcribed ASR')
    parser_b = subparsers.add_parser(
        'load', help='Load wav transcriptions and reference with full paths to wavs')
    parser_b.add_argument(
        'reference', help='Key value file: Keys contain paths to wav files. Values are reference transcriptions.')

    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        if not args.f:
            print "\nThe directory '%s' already exists!\n" % args.out_dir
            parser.print_usage()
            parser.exit()
    else:
        # create the dir
        try:
            os.makedirs(args.out_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST or os.path.isdir(args.out_dir):
                raise exc

    cfg = Config.load_configs(args.configs, use_default=True)

    if args.command == 'extract':
        extract_from_xml(args.indomain_data_dir, args.out_dir, cfg)
    elif args.command == 'load':
        decode_with_reference(args.reference, args.out_dir, cfg)
    else:
        raise Exception('Argparse mechanism failed: Should never happen')
