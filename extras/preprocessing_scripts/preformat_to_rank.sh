#!/bin/sh
#README

#call me
#sh format_multi_class_files.sh obj_state.log partition_factor

INPUT=$1
PARTITIONS=$2

if [ "$PARTITIONS" -lt 1 ]; then
  echo "minimum number of partitions must be 0"
  exit -1
fi

START_COLUMN=3
END_COLUMN=15

DATASET=/tmp/dataset.data

rm -rf $DATASET

awk -F, -v start_col=3 -v end_col=15 -v temp=0 -v DATASET=$DATASET ' \
    func write_to_file(new_lines,file_name) { \
      printf "%s", new_lines >> (file_name); \
      close(file_name); \
    } \
    BEGIN { \
      getline; \
    } \
    { \
        activity=0; \
        for(i=5;i<=7;i++) { \
          activity = activity + $i; \
        } \
        new_line = activity "\t"; \
        for(i=start_col;i<=(end_col-1);i++) { \
          if ( (i!=5)  && (i!=6)  && (i!=7) ) { \
            new_line = new_line $i "\t"; \
          } \
        }; \
        label=$(end_col); \
        if ( $(end_col)==3 ) { \
          label=2; \
        } \
        if ( $(end_col)==7 ) { \
          label=3; \
        } \
        new_line_ready = new_line label "\n"; \
        write_to_file(new_line_ready,DATASET); \
    }' $INPUT


if [ "$PARTITIONS" -gt 1 ]; then
  echo "linear greater than 1: $PARTITIONS"
  sh /home/guthemberg/Documents/workplace/pcp//extras/preprocessing_scripts/split_nfolds.sh $DATASET $PARTITIONS
  cp /tmp/f1.data /tmp/train.data
  cp /tmp/f2.data /tmp/test.data
else
  echo "linear NOT greater than 1: $PARTITIONS"
  sh /home/guthemberg/Documents/workplace/pcp//extras/preprocessing_scripts/split_nfolds.sh $DATASET 1
  cp /tmp/f1.data /tmp/train.data
  cp /tmp/f1.data /tmp/test.data
fi  
#for computing nDCG, test data must be sorted
/bin/bash -c "sort -k11 -t$'\t' -n -r /tmp/test.data > /tmp/test.data.sorted"

