#!/bin/bash
#
# Given the pronouncing dictionary, converts it into the form we'll be 
# using with HTK.
#
# Also adds in extra words we discovered we needed.
#

# The dictionaries are already in the HTK format, so just merge them.

perl $TRAIN_SCRIPTS/MergeDict.pl $TRAIN_COMMON/csdict $TRAIN_COMMON/csdict.ext > $WORK_DIR/dict_full
