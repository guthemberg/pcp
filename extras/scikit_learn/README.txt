how to run (draft)

 1321  man awk
 1322  history |grep BEST
 1323  sh ~/Documents/workplace/pcp//extras/preprocessing_scripts/format_and_split_nfolds.sh /home/guthemberg/obj_state.log_THE_BEST_ONE 2
 1324  history |grep validate
 1325  history |grep once
 1326  python ~/Documents/workplace/pcp/extras/scikit_learn/svm.py once /tmp/train.data_one_class /tmp/test.data_one_class
 1327  sh ~/Documents/workplace/pcp//extras/preprocessing_scripts/format_and_split_nfolds.sh /home/guthemberg/obj_state.log_THE_BEST_ONE 80
 1328  less ~/Documents/workplace/pcp//extras/preprocessing_scripts/format_and_split_nfolds.sh
 1329  ps xau|grep python
 1330  kill -9 589


then run pilot for listening to predictions requests

  146  ps xau|grep pilot
  147  kill -9 31191 && rm -rf /var/run/pilotd.pid 
  148  /home/guthemberg/Documents/workplace/pilot/pilotd start
  149  ps xau|grep pilot
  150  kill -9 6305 && rm -rf /var/run/pilotd.pid 
  151  /home/guthemberg/Documents/workplace/pilot/pilotd start
  152  ps xau|grep pilot

