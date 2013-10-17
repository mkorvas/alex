#!/bin/bash
# Make the MLF for the training data.
#
# After we do prep_train.sh, we want to create a word level MLF for all
# the files that have been successfully converted to MFC files.

# Create a file listing all the MFC files in the train directory.
find $TRAIN_DATA -iname '*.mfc' > $WORK_DIR/train_mfc_files.txt

# Now create the MLF file using a script. We prune anything that
# has words that aren't in our dictionary, producing a MLF with only
# these files and a corresponding script file.
if [[ $1 != "prune" ]]
then
  python $TRAIN_SCRIPTS/CreateMLF.py "-"                 $WORK_DIR/train_words.mlf $WORK_DIR/train.scp $TRAIN_DATA $TRAIN_DATA_SOURCE'/*.trn' > $LOG_DIR/train_missing_words.log
else
  python $TRAIN_SCRIPTS/CreateMLF.py $WORK_DIR/dict_full $WORK_DIR/train_words.mlf $WORK_DIR/train.scp $TRAIN_DATA $TRAIN_DATA_SOURCE'/*.trn' > $LOG_DIR/train_missing_words.log
fi

if [[ -n "$2" ]]
then
	echo -e "$2+1,\$d\nwq" | ed -s $WORK_DIR/train.scp 2>/dev/null \
		|| true  # ed fails iff the file has less than "$2" lines
						 # That is nothing wrong, so we ignore it silently.
fi
