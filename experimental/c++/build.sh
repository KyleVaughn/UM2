INT_OPTIONS="-fiopenmp -fopenmp-targets=spir64"
GNU_OPTIONs="-m64" 

INCLUDES="-I${MKLROOT}/include -I/usr/local/include/eigen3"

GNU_WARNINGS="-Wpedantic -Wall -Wextra"
INT_WARNINGS="-pedantic -Wall -Wextra"

INT_OPT_FLAGS="-O3" #"-fast" # Gives -ipo, -O3, -no-prec-div, -static, -fp-model fast=2, and -xHost
GNU_OPT_FLAGS="-Ofast -march=native" 

INT_DYNAMIC_MKL="-fiopenmp -fopenmp-targets=spir64 -fsycl -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lsycl -lOpenCL -lstdc++ -lpthread -lm -ldl"
GNU_DYNAMIC_MKL="                                         -L${MKLROOT}/lib/intel64            -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5                          -lpthread -lm -ldl"

INT_STATIC_MKL="-fiopenmp -fopenmp-targets=spir64  -fsycl  ${MKLROOT}/lib/intel64/libmkl_sycl.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lsycl -lOpenCL -lstdc++ -lpthread -lm -ldl"
GNU_STATIC_MKL="                                                                                -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5                          -lpthread -lm -ldl"

icpx  main.cpp -o main.exe ${INT_OPTIONS} ${INCLUDES} ${INT_WARNINGS} ${INT_OPT_FLAGS} ${INT_STATIC_MKL} && ./main.exe
g++-12   main.cpp -o main.exe ${GNU_OPTIONS} ${INCLUDES} ${GNU_WARNINGS} ${GNU_OPT_FLAGS} ${GNU_STATIC_MKL} && ./main.exe
