#!/bin/bash
# Make the MLF for the test data.
#
# After we do prep_test.sh, we want to create a word level MLF for all
# the files that have been successfully converted to MFC files.

# Create a file listing all the MFC files in the test directory
find $TEST_DATA -iname '*.mfc' > $WORK_DIR/test_mfc_files.txt

# Now create the MLF file using a script, we prune out anything that
# has words that aren't in our dictionary, producing a MLF with only
# these files and a corresponding script file.
if [[ $1 != "prune" ]]
then
  python $TRAIN_SCRIPTS/CreateMLF.py "-"                 $WORK_DIR/test_words.mlf $WORK_DIR/test.scp $TEST_DATA $TEST_DATA_SOURCE'/*.trn' > $LOG_DIR/test_missing_words.log
else
  python $TRAIN_SCRIPTS/CreateMLF.py $WORK_DIR/dict_full $WORK_DIR/test_words.mlf $WORK_DIR/test.scp $TEST_DATA $TEST_DATA_SOURCE'/*.trn' > $LOG_DIR/test_missing_words.log
fi

if [[ -n "$2" ]]
then
	echo -e "$2+1,\$d\nwq" | ed -s $WORK_DIR/test.scp 2>/dev/null \
		|| true  # ed fails iff the file has less than "$2" lines
						 # That is nothing wrong, so we ignore it silently.
fi
