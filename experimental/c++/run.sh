FILES=("vector")    
for FILE in $FILES    
do    
  echo "$OPT_LEVEL"    
  g++ ./benchmarks/$FILE.cpp -o $FILE $(< gnu_flags) && ./$FILE
done    
