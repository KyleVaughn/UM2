for OPT_LEVEL in -O0 -O1 -O2 -O3
do
    echo "$OPT_LEVEL"
    julia $OPT_LEVEL ./benchmarks/vector.jl
done
