#!/bin/bash
# Builds the word list and network we need for recognition

cd $WORK_DIR

# Get word list
python $TRAIN_SCRIPTS/CreateWordList.py $WORK_DIR/dict_full \
	$TRAIN_DATA_SOURCE'/*.trn' \
	| LC_ALL=C sort \
	| LC_ALL=C uniq \
	| grep -v "(" > $WORK_DIR/word_list_train
python $TRAIN_SCRIPTS/CreateWordList.py $WORK_DIR/dict_full \
	$TEST_DATA_SOURCE'/*.trn' \
	| LC_ALL=C sort \
	| LC_ALL=C uniq \
	| grep -v "(" > $WORK_DIR/word_list_test


# NOTE This should be run just once, even in repeated runs.
# We need sentence start and end symbols which match the WSJ
# standard language model and produce no output symbols.
cat <<END |
<s>	[] sil
</s>	[] sil
silence	sil
_INHALE_	_inhale_
_LAUGH_	_laugh_
_EHM_HMM_	_ehm_hmm_
_NOISE_	_noise_
_SIL_	sil
END
tee $WORK_DIR/dict_train |
tee $WORK_DIR/dict_test  |
tee -a $WORK_DIR/dict_full


# Add pronunciations for each word
perl $TRAIN_SCRIPTS/WordsToDictionary.pl $WORK_DIR/word_list_train \
	$WORK_DIR/dict_full $TEMP_DIR/dict_train
cat $TEMP_DIR/dict_train >> $WORK_DIR/dict_train
perl $TRAIN_SCRIPTS/WordsToDictionary.pl $WORK_DIR/word_list_test \
	$WORK_DIR/dict_full $TEMP_DIR/dict_test
cat $TEMP_DIR/dict_test >> $WORK_DIR/dict_test

# Create a dictionary with a sp short pause after each word, this is so 
# when we do the phone alignment from the word level MLF, we get the sp 
# phone in between the words.  This version duplicates each entry and uses 
# a long pause sil after each word as well.
perl $TRAIN_SCRIPTS/AddSp.pl $WORK_DIR/dict_full 1 > $WORK_DIR/dict_full_sp_sil
perl $TRAIN_SCRIPTS/AddSp.pl $WORK_DIR/dict_train 1 > $WORK_DIR/dict_train_sp_sil
perl $TRAIN_SCRIPTS/AddSp.pl $WORK_DIR/dict_test 1 > $WORK_DIR/dict_test_sp_sil

# Add to the word list non-speech events
echo "_INHALE_" >> $WORK_DIR/word_list_test
echo "_LAUGH_" >> $WORK_DIR/word_list_test
echo "_EHM_HMM_" >> $WORK_DIR/word_list_test
echo "_NOISE_" >> $WORK_DIR/word_list_test

# Build the word network as a word loop of words in the testing data.
HBuild -A -T 1 -u '<UNK>' -s '<s>' '</s>' $WORK_DIR/word_list_test $WORK_DIR/wdnet_zerogram > $LOG_DIR/hbuild.log

if [ -f $DATA_SOURCE_DIR/wdnet_bigram ]
then
  cp $DATA_SOURCE_DIR/wdnet_bigram $WORK_DIR/wdnet_bigram
fi

if [ -f $DATA_SOURCE_DIR/arpa_trigram ]
then
  cp $DATA_SOURCE_DIR/arpa_trigram $WORK_DIR/arpa_trigram
fi
if [ -f $WORK_DIR/arpa_trigram ]
then
	python $TRAIN_SCRIPTS/WordListFromARPALM.py $WORK_DIR/arpa_trigram > $WORK_DIR/word_list_hdecode
fi
if [ -f $WORK_DIR/dict_hdecode ]
then
  perl $TRAIN_SCRIPTS/WordsToDictionary.pl $WORK_DIR/word_list_hdecode $WORK_DIR/dict_full $WORK_DIR/dict_hdecode
fi
