#!/bin/bash

# Trains up triphone models for English.

# TODO Evaluate on dev instead of test.

# Stop on errors.
set -e

source env_voip_en.sh

date

echo "Training word internal triphone model for English"
echo ""
echo "Environment variables:"
echo "TRAIN_COMMON       = $TRAIN_COMMON"
echo "TRAIN_SCRIPTS      = $TRAIN_SCRIPTS"
echo "HEREST_SPLIT       = $HEREST_SPLIT"
echo "HEREST_THREADS     = $HEREST_THREADS"
echo "DEC_N_PROCS        = $DEC_N_PROCS"
echo ""
echo "TRAIN_DATA_SOURCE  = $TRAIN_DATA_SOURCE"
echo "TEST_DATA_SOURCE   = $TEST_DATA_SOURCE"
echo ""
echo "WORK_DIR           = $WORK_DIR"
echo "TEMP_DIR           = $TEMP_DIR"
echo "LOG_DIR            = $LOG_DIR"
echo "TRAIN_DATA         = $TRAIN_DATA"
echo "TEST_DATA          = $TEST_DATA"
echo ""
echo "N_TRAIN_FILES      = $N_TRAIN_FILES"
echo "N_TEST_FILES       = $N_TEST_FILES"
echo ""
echo "RO                 = $RO"
echo "TB                 = $TB"
echo "IP                 = $IP"
echo "SFZ                = $SFZ"
echo "SFB                = $SFB"
echo "SFT                = $SFT"
echo ""
echo "CROSS              = $CROSS"
echo ""

cd $WORK_DIR

# We need to massage the CMU dictionary for our use.
echo "Preparing CMU English dictionary..."
$TRAIN_SCRIPTS/prep_cmu_dict.sh

# Code the audio files to MFCC feature vectors.
# NOTE: Skip this step in subsequent runs.
echo "Coding test audio..."
$TRAIN_SCRIPTS/prep_param_test.sh

echo "Coding train audio..."
$TRAIN_SCRIPTS/prep_param_train.sh

# Initial setup of the language model, dictionary, training and test MLFs.
echo "Building unigram language models and dictionary..."
$TRAIN_SCRIPTS/build_lm_en.sh
echo "Building training MLF..."
$TRAIN_SCRIPTS/make_mlf_train.sh prune "$N_TRAIN_FILES"
echo "Building test MLF..."
$TRAIN_SCRIPTS/make_mlf_test.sh prune "$N_TEST_FILES"

date


##############
#  TRAINING  #
##############

# Get the basic monophone models trained.
echo "Flat starting monophones..."
$TRAIN_SCRIPTS/flat_start.sh

# Create a new MLF that is aligned based on our monophone model.
echo "Aligning with monophones..."
$TRAIN_SCRIPTS/align_mlf.sh

# More training for the monophones, create triphones, train
# triphones, tie the triphones, train tied triphones, then
# mix up the number of Gaussians per state.
echo "Training monophones..."
$TRAIN_SCRIPTS/train_mono.sh
echo "Prepping triphones..."
$TRAIN_SCRIPTS/prep_tri.sh $CROSS
echo "Training triphones..."
$TRAIN_SCRIPTS/train_tri.sh

# These values of RO and TB seem to work fairly well, but
# there may be more optimal values.
echo "Prepping state-tied triphones..."
$TRAIN_SCRIPTS/prep_tied.sh $RO $TB $CROSS

echo "Training state-tied triphones..."
$TRAIN_SCRIPTS/train_tied.sh
echo "Mixing up..."
$TRAIN_SCRIPTS/train_mixup.sh

date

# Re-align the training data with the best triphone models.
# MK: What does this help with when no re-estimation is done?
echo "Aligning with triphones..."
$TRAIN_SCRIPTS/realign.sh hmm67 tiedlist

echo "End of training"

#############
#  TESTING  #
#############

# Evaluate how we did with the zerogram language model.
#
# Decoding with a zerogram language model cannot be done with cross-word
# triphone context.
echo "Decoding zerogram language model"
$TRAIN_SCRIPTS/eval_test_no_lat.sh hmm42 ro"$RO"_tb"$TB"_prune350_zerogram_08 350.0 $IP $SFZ $WORK_DIR/wdnet_zerogram $WORK_DIR/dict_test_sp_sil wit &
[[ $((1 % $DEC_N_PROCS)) -eq 0 ]] && wait
$TRAIN_SCRIPTS/eval_test_no_lat.sh hmm47 ro"$RO"_tb"$TB"_prune350_zerogram_10 350.0 $IP $SFZ $WORK_DIR/wdnet_zerogram $WORK_DIR/dict_test_sp_sil wit &
[[ $((2 % $DEC_N_PROCS)) -eq 0 ]] && wait
$TRAIN_SCRIPTS/eval_test_no_lat.sh hmm52 ro"$RO"_tb"$TB"_prune350_zerogram_12 350.0 $IP $SFZ $WORK_DIR/wdnet_zerogram $WORK_DIR/dict_test_sp_sil wit &
[[ $((3 % $DEC_N_PROCS)) -eq 0 ]] && wait
$TRAIN_SCRIPTS/eval_test_no_lat.sh hmm57 ro"$RO"_tb"$TB"_prune350_zerogram_14 350.0 $IP $SFZ $WORK_DIR/wdnet_zerogram $WORK_DIR/dict_test_sp_sil wit &
[[ $((4 % $DEC_N_PROCS)) -eq 0 ]] && wait
$TRAIN_SCRIPTS/eval_test_no_lat.sh hmm62 ro"$RO"_tb"$TB"_prune350_zerogram_16 350.0 $IP $SFZ $WORK_DIR/wdnet_zerogram $WORK_DIR/dict_test_sp_sil wit &
[[ $((5 % $DEC_N_PROCS)) -eq 0 ]] && wait
$TRAIN_SCRIPTS/eval_test_no_lat.sh hmm67 ro"$RO"_tb"$TB"_prune350_zerogram_18 350.0 $IP $SFZ $WORK_DIR/wdnet_zerogram $WORK_DIR/dict_test_sp_sil wit &
wait

date

# Evaluate how we did with the bigram language model if it is available.
if [ -f $WORK_DIR/wdnet_bigram ]
then
  echo "Decoding bigram language model"
  $TRAIN_SCRIPTS/eval_test_no_lat.sh hmm42 ro"$RO"_tb"$TB"_prune350_bigram_08 350.0 $IP $SFB $WORK_DIR/wdnet_bigram $WORK_DIR/dict_full_sp_sil $CROSS &
	[[ $((1 % $DEC_N_PROCS)) -eq 0 ]] && wait
  $TRAIN_SCRIPTS/eval_test_no_lat.sh hmm47 ro"$RO"_tb"$TB"_prune350_bigram_10 350.0 $IP $SFB $WORK_DIR/wdnet_bigram $WORK_DIR/dict_full_sp_sil $CROSS &
	[[ $((2 % $DEC_N_PROCS)) -eq 0 ]] && wait
  $TRAIN_SCRIPTS/eval_test_no_lat.sh hmm52 ro"$RO"_tb"$TB"_prune350_bigram_12 350.0 $IP $SFB $WORK_DIR/wdnet_bigram $WORK_DIR/dict_full_sp_sil $CROSS &
	[[ $((3 % $DEC_N_PROCS)) -eq 0 ]] && wait
  $TRAIN_SCRIPTS/eval_test_no_lat.sh hmm57 ro"$RO"_tb"$TB"_prune350_bigram_14 350.0 $IP $SFB $WORK_DIR/wdnet_bigram $WORK_DIR/dict_full_sp_sil $CROSS &
	[[ $((4 % $DEC_N_PROCS)) -eq 0 ]] && wait
  $TRAIN_SCRIPTS/eval_test_no_lat.sh hmm62 ro"$RO"_tb"$TB"_prune350_bigram_16 350.0 $IP $SFB $WORK_DIR/wdnet_bigram $WORK_DIR/dict_full_sp_sil $CROSS &
	[[ $((5 % $DEC_N_PROCS)) -eq 0 ]] && wait
  $TRAIN_SCRIPTS/eval_test_no_lat.sh hmm67 ro"$RO"_tb"$TB"_prune350_bigram_18 350.0 $IP $SFB $WORK_DIR/wdnet_bigram $WORK_DIR/dict_full_sp_sil $CROSS &
  wait
fi

date

# Evaluate how we did with the trigram language model if it is available.
# The trigram LM is only decoded with cross-boundary triphones.
#
# XXX For this to work, it is apparently necessary to remove superfluous 
# words that translate to "sil" from the pronouncing dictionary 
# (model_voip_en/dict_hdecode), otherwise you will get the following error:
# ERROR [+9999]  HLVNet: no model label for phone (ao-sil+ah)
#
# Still, after these alternative words that translate to sil have been 
# removed, the script does not work, complaining with the following 
# message:
# ERROR [+9999]  AssignWEIds: too many pronunciations for PronId/LMId type.  
# According to http://speech-research.com/htkSearch/index.php?ID=249570,
# this is caused by the HTK compiled with 16bit representation for word 
# pronunciations (or their IDs). To fix this, you might need to recompile 
# HTK using 32bit representations (int).
if [ -f $WORK_DIR/arpa_trigram ]
then
  echo "Decoding trigram language model"
  $TRAIN_SCRIPTS/eval_test_hd_no_lat.sh hmm42 ro"$RO"_tb"$TB"_prune150_trigram_08 150.0 $IP $SFT $WORK_DIR/arpa_trigram $WORK_DIR/dict_hdecode cross &
	[[ $((1 % $DEC_N_PROCS)) -eq 0 ]] && wait
  $TRAIN_SCRIPTS/eval_test_hd_no_lat.sh hmm47 ro"$RO"_tb"$TB"_prune150_trigram_10 150.0 $IP $SFT $WORK_DIR/arpa_trigram $WORK_DIR/dict_hdecode cross &
	[[ $((2 % $DEC_N_PROCS)) -eq 0 ]] && wait
  $TRAIN_SCRIPTS/eval_test_hd_no_lat.sh hmm52 ro"$RO"_tb"$TB"_prune150_trigram_12 150.0 $IP $SFT $WORK_DIR/arpa_trigram $WORK_DIR/dict_hdecode cross &
	[[ $((3 % $DEC_N_PROCS)) -eq 0 ]] && wait
  $TRAIN_SCRIPTS/eval_test_hd_no_lat.sh hmm57 ro"$RO"_tb"$TB"_prune150_trigram_14 150.0 $IP $SFT $WORK_DIR/arpa_trigram $WORK_DIR/dict_hdecode cross &
	[[ $((4 % $DEC_N_PROCS)) -eq 0 ]] && wait
  $TRAIN_SCRIPTS/eval_test_hd_no_lat.sh hmm62 ro"$RO"_tb"$TB"_prune150_trigram_16 150.0 $IP $SFT $WORK_DIR/arpa_trigram $WORK_DIR/dict_hdecode cross &
	[[ $((5 % $DEC_N_PROCS)) -eq 0 ]] && wait
  $TRAIN_SCRIPTS/eval_test_hd_no_lat.sh hmm67 ro"$RO"_tb"$TB"_prune150_trigram_18 150.0 $IP $SFT $WORK_DIR/arpa_trigram $WORK_DIR/dict_hdecode cross &
  wait
fi

date

# TODO We might want to take the best model and export it, not necessarily 
# the last trained one.

$TRAIN_SCRIPTS/export_models.sh hmm63 text

date
