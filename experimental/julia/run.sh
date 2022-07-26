FILES=("vector")
for OPT_LEVEL in -O3
do
  for FILE in $FILES
  do
    echo "$OPT_LEVEL"
    julia $OPT_LEVEL ./benchmarks/$FILE.jl
  done
done
