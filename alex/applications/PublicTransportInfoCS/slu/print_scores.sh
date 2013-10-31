echo "DATA ALL ASR - ASR scores"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/asrscore.py all.trn all.asr

echo "=========================================================================================================="
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA ALL ASR - HDC SLU"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py all.trn.hdc.sem all.asr.hdc.sem
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA ALL ASR - ASR model"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py all.trn.hdc.sem all.asr.model.asr.sem.out
echo
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA ALL NBL - HDC SLU"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py all.trn.hdc.sem all.nbl.hdc.sem
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA ALL NBL - NBL model"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py all.trn.hdc.sem all.nbl.model.nbl.sem.out
echo
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA ALL TRN - HDC SLU"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py all.trn.hdc.sem all.trn.hdc.sem
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA ALL TRN - TRN model"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py all.trn.hdc.sem all.trn.model.trn.sem.out
echo
echo "=========================================================================================================="
echo


echo "----------------------------------------------------------------------------------------------------------"
echo "DATA DEV ASR - HDC SLU"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py dev.trn.hdc.sem dev.asr.hdc.sem
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA DEV ASR - ASR model"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py dev.trn.hdc.sem dev.asr.model.asr.sem.out
echo
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA DEV NBL - HDC SLU"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py dev.trn.hdc.sem dev.nbl.hdc.sem
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA DEV NBL - NBL model"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py dev.trn.hdc.sem dev.nbl.model.nbl.sem.out
echo
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA DEV TRN - HDC SLU"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py dev.trn.hdc.sem dev.trn.hdc.sem
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA DEV TRN - TRN model"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py dev.trn.hdc.sem dev.trn.model.trn.sem.out
echo
echo "=========================================================================================================="
echo

echo "----------------------------------------------------------------------------------------------------------"
echo "DATA TEST ASR - HDC SLU"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py test.trn.hdc.sem test.asr.hdc.sem
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA TEST ASR - ASR model"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py test.trn.hdc.sem test.asr.model.asr.sem.out
echo
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA TEST NBL - HDC SLU"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py test.trn.hdc.sem test.nbl.hdc.sem
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA TEST NBL - NBL model"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py test.trn.hdc.sem test.nbl.model.nbl.sem.out
echo
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA TEST TRN - HDC SLU"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py test.trn.hdc.sem test.trn.hdc.sem
echo "----------------------------------------------------------------------------------------------------------"
echo "DATA TEST TRN - TRN model"
echo "----------------------------------------------------------------------------------------------------------"
../../../corpustools/semscore.py test.trn.hdc.sem test.trn.model.trn.sem.out
echo