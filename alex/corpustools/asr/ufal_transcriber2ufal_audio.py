#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import argparse
import codecs
import collections
import os.path
import random
import re
import subprocess
import xml.dom.minidom


"""
This program process transcribed audio in Transcriber files and copies all
relevant speech segments into a destination directory.
It also extracts transcriptions and saves them alongside the copied wavs.

It scans for '*.trs' to extract transcriptions and names of wave files.

########################################################################
# An example ignore list file could contain the following three lines: #
########################################################################

/home/matej/wc/vystadial/data/call-logs/voip-0012097903339-121001_014358/jurcic-010-121001_014624_0014528_0014655.trs
jurcic-006-121001_014526_0008608_0008758.trs
jurcic-??[13579]*.trs

The first one is an example of an ignored path. On UNIX, it has to start with
a slash. On other platforms, an analogic convention has to be used.
#
The second one is an example of a literal glob.
#
The last one is an example of a more advanced glob. It says basically that
all odd dialogue turns should be ignored.

"""

_subst = [('<SILENCE>', '_SIL_'),
          ('<INHALE>', '_INHALE_'),
          ('<NOISE>', '_NOISE_'),
          ('<COUGH>', '_EHM_HMM_'),
          ('<MOUTH>', '_EHM_HMM_'),
          ('<LAUGH>', '_LAUGH_'),
          ('<EHM A>', '_EHM_HMM_'),
          ('<EHM N>', '_EHM_HMM_'),
          ('<EHM *>', '_EHM_HMM_'),
          (ur'\([^(]*\(([^)]*)\)\)', '\\2'),  # (written form (pronounced))
            # NOTE '\\2' is used instead of '\\1' since '\\1' will refer to the
            # preceding 0 or 1 characters when `_subst' is used next
          ('JESLTI', 'JESTLI'),
          (u'NMŮŽU', u'NEMŮŽU'),
          ('6E', ' '),   # XXX What is this??
          ]
for idx, tup in enumerate(_subst):
    pat, sub = tup
    _subst[idx] = (re.compile(ur'(\W|^){pat}(?=$|\W)'.format(pat=pat)),
                   r'\1' + sub)
_subst.append((re.compile('\\^'), ''))

_hesitation = [re.compile(r"\bUMMM\b")]

_excluded_characters = ['|', '-', '=', '(', ')', '[', ']', '{', '}', '<', '>',
                       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
_more_spaces = re.compile(r'\s{2,}')
_sure_punct_rx = re.compile(r'[.?!",_]')


def normalise_trs(text):
    text = _sure_punct_rx.sub(' ', text)
    text = text.strip().upper()
    # Do dictionary substitutions.
    for pat, sub in _subst:
        text = pat.sub(sub, text)
    text = _more_spaces.sub(' ', text).strip()
    for word in _hesitation:
        text = word.sub('(HESITATION)', text)
    # remove signs of (1) incorrect pronunciation, (2) stuttering, (3) bargein
    for char in '*+~':
        text = text.replace(char, '')
    return text


def exclude(text):
    """
    Determines whether `text' is not good enough and should be excluded. "Good
    enough" is defined as containing none of `_excluded_characters' and being
    longer than one word.

    """
    for char in _excluded_characters:
        if char in text:
            return True

    if len(text) < 2:
        return True

    return False


def unique_str():
    """Generates a fairly unique string."""
    return hex(random.randint(0, 256 * 256 * 256 * 256 - 1))[2:]


def cut_wavs(src, tgt, start, end):
    """Cuts out the interval `start'--`end' from the wav file `src' and saves
    it to `tgt'.

    """
    existed = os.path.exists(tgt)
    cmd = ("sox", "--ignore-length", src, tgt,
           "trim", str(start), str(end - start))
    print " ".join(cmd)
    subprocess.call(cmd)
    return existed


def save_transcription(trs_fname, trs):
    """
    Echoes `trs' into `trs_fname'. Returns True iff the
    output file already existed.

    """
    existed = os.path.exists(trs_fname)
    if not trs.endswith('\n'):
        trs += '\n'
    with codecs.open(trs_fname, 'w+', encoding='UTF-8') as trs_file:
        trs_file.write(trs)
        # trs_file.write(trs.encode('ascii', 'ignore'))
    return existed


def extract_wavs_trns(_file, outdir, trs_only=False, verbose=False):
    """Extracts wavs and their transcriptions from the provided big wav and the
    transcriber file.

    """

    # Parse the file.
    doc = xml.dom.minidom.parse(_file)
    uturns = doc.getElementsByTagName("Sync")

    size = 0
    n_overwrites = 0
    n_missing_wav = 0
    n_missing_trs = 0
    for uturn in uturns:
        if verbose:
            print '-' * (getTerminalSize()[1] or 120)

        # Retrieve the user turn's data.
        starttime = float(uturn.getAttribute('time').strip())
        if uturn.nextSibling.nodeType == uturn.TEXT_NODE:
            transcription = uturn.nextSibling.data.strip()
        else:
            transcription = ''
            n_missing_trs += 1
        try:
            endtime = float(
                uturn.nextSibling.nextSibling.getAttribute('time').strip())
        except:
            endtime = 9999.000

        # Construct various involved file names.
        src_wav_fname = _file.replace('.trs', '.wav')
        tgt_ext = '-{start:07.2f}-{end:07.2f}-{hash}.wav'.format(
            start=starttime, end=endtime, hash=unique_str())
        tgt_wav_fname = os.path.join(
            outdir, os.path.basename(_file).replace('.trs', tgt_ext))
        transcription_file_name = tgt_wav_fname + '.trn'

        if verbose:
            print " #f: {tgt}; # s: {start}; # e: {end}; t: {trs}".format(
                tgt=os.path.basename(tgt_wav_fname), start=starttime,
                end=endtime, trs=transcription)

        # Normalise
        transcription = normalise_trs(transcription)
        if verbose:
            print "  after normalisation:", transcription
        if exclude(transcription):
            if verbose:
                print "  ...excluded"
            continue

        # Save the transcription and corresponding wav files.
        wc.update(transcription.split())
        n_overwrites += save_transcription(transcription_file_name,
                                           transcription)
        if not trs_only:
            try:
                cut_wavs(src_wav_fname, tgt_wav_fname, starttime, endtime)
                size += os.path.getsize(tgt_wav_fname)
            except OSError:
                n_missing_wav += 1
                print "Missing audio file: ", tgt_wav_fname

    return size, n_overwrites, n_missing_wav, n_missing_trs


def convert(args):
    # TODO docstring
    # Unpack the arguments.
    infname = args.infname
    outdir = args.outdir
    verbose = args.verbose
    trs_only = args.only_transcriptions
    ignore_list_file = args.ignore
    # Read in the ignore list.
    ignore_paths = set()
    ignore_globs = set()
    if ignore_list_file:
        for path_or_glob in ignore_list_file:
            path_or_glob = path_or_glob.rstrip('\n')
            # For lines that list absolute paths,
            if os.path.abspath(path_or_glob) == os.path.normpath(path_or_glob):
                # add them to the list of paths to ignore.
                ignore_paths.add(path_or_glob)
            # For other lines, treat them as basename globs.
            else:
                ignore_globs.add(path_or_glob)
        ignore_list_file.close()

    # Get all but the ignored transcriptions.
    if os.path.isdir(infname):
        trs_paths = find(infname, '*.trs',
                         ignore_globs=ignore_globs, ignore_paths=ignore_paths)
    else:
        trs_paths = list()
        with open(infname, 'r') as inlist:
            for line in inlist:
                trs_paths.extend(find(line.strip(), '*.trs',
                                      mindepth=1, maxdepth=1,
                                      ignore_globs=ignore_globs,
                                      ignore_paths=ignore_paths))

    size = 0
    n_overwrites = 0
    n_missing_wav = 0
    n_missing_trs = 0
    for trs_path in trs_paths:

        if verbose:
            print "Processing transcription file: ", trs_path

        cursize, cur_n_overwrites, cur_n_missing_wav, cur_n_missing_trs = \
            extract_wavs_trns(trs_path, outdir, trs_only, verbose)
        size += cursize
        n_overwrites += cur_n_overwrites
        n_missing_wav += cur_n_missing_wav
        n_missing_trs += cur_n_missing_trs

    print "Size of copied audio data:", size

    sec = size / 16000
    hour = sec / 3600.0

    print "Length of audio data in hours (for 8kHz 16bit WAVs):", hour
    # Return the number of file collisions and overwrites.
    return n_overwrites, n_missing_wav, n_missing_trs


if __name__ == '__main__':
    wc = collections.Counter()  # word counter

    import sys

    import autopath
    from alex.utils.fs import find
    from alex.utils.ui import getTerminalSize

    random.seed(1)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
        This program process transcribed audio in Transcriber (*.trs) files and
        copies all relevant speech segments into a destination directory.
        It also extracts transcriptions and saves them alongside the copied
        wavs.

        It scans for '*.trs' to extract transcriptions and names of wave files.

      """)

    parser.add_argument('infname',
                        action="store",
                        help='an input directory with trs and wav files, or '
                             'a file listing the trs files')
    parser.add_argument('outdir',
                        action="store",
                        help='an output directory for files with audio and '
                             'their transcription')
    parser.add_argument('-i', '--ignore',
                        type=argparse.FileType('r'),
                        metavar='FILE',
                        help='Path towards a file listing globs of wav files '
                             'that should be ignored.  The globs are '
                             'interpreted wrt. the current working directory. '
                             'For an example, see the source code.')
    parser.add_argument('-t', '--only-transcriptions',
                        action="store_true",
                        help='only normalise transcriptions, ignore audio '
                             'files')
    parser.add_argument('-v', '--verbose',
                        action="store_true",
                        help='set verbose output')
    parser.add_argument('-w', '--word-list',
                        default='word_list',
                        metavar='FILE',
                        help='Path towards an output file to contain a list '
                             'of words that appeared in the transcriptions, '
                             'one word per line.')

    args = parser.parse_args()

    # Ensure we can print unicode.
    if not sys.stdout.isatty():
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    if not sys.stderr.isatty():
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

    # Do the copying.
    if args.verbose:
        escape_value = (lambda value: unicode(value).replace('\\', '\\\\')
                        .replace('"', '\\"'))
        print '*' * (getTerminalSize()[1] or 120)
        print "The program was called with these arguments:"
        print "\n".join('  {key}: "{value}"'.format(key=key,
                                                    value=escape_value(value))
                        for key, value in args._get_kwargs())
        print '*' * (getTerminalSize()[1] or 120)
    n_overwrites, n_missing_wav, n_missing_trs = convert(args)

    # Report.
    msg = ("# overwrites: {ovw};  # without transcription: {wotrs};  "
           "# missing: {msng}").format(ovw=n_overwrites, wotrs=n_missing_trs,
                                       msng=n_missing_wav)
    print msg

    with codecs.open(args.word_list, 'w', "utf-8") as word_list_file:
        for word in sorted(wc):
            word_list_file.write("{word}\t{count}\n".format(
                word=word, count=wc[word]))
