#this script splits input data file n folds. how to call sh spli_nfolds.sh input.data_full_path n
#files are created in /tmp/f*.data
#
#EXAMPLE
##split content_popularity.data into 10 files from /tmp/f1.data to /tmp/f10.data
#
##sh split_nfolds.sh content_popularity.data 10

PREFIX=/tmp/f
rm -rf ${PREFIX}*
awk -F',' -v prefix=${PREFIX} -v suffix=".data" -v counter=1 -v n=$2 ' func write_to_file(line,file) { printf("%s\n",$0 ) >> (file); close(file); } BEGIN {getline;} { \
  filename = (prefix counter suffix); \
  write_to_file($0,filename); \
  counter = counter + 1; \
  if (counter>n) { \
    counter=1; \
  } \
} END {} ' $1

