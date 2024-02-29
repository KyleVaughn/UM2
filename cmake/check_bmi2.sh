#!/bin/bash

# BMI2 is supported by AMD's Excavator architecture and later, but isn't performant
# until Zen 3. This script checks for the AMD architectures which support BMI2, but
# but where it may not be performant: Excavator, Zen 1, and Zen 2

# Usage: ./check_bmi2.sh CXX_COMPILER
# Example: ./check_bmi2.sh clang++

# Check that a compiler is provided
if [ -z "$1" ]; then
    echo "No compiler provided. Usage: ./check_bmi2.sh CXX_COMPILER"
    exit 1
fi

$1 -### -E - -march=native &> arch.txt

# Excavator
if grep -q bdver4 arch.txt; then
    echo "AMD Excavator architecture detected. BMI2 instructions may not be performant."
fi

# Zen 1
if grep -q znver1 arch.txt; then
    echo "AMD Zen 1 architecture detected. BMI2 instructions may not be performant."
fi

# Zen 2
if grep -q znver2 arch.txt; then
    echo "AMD Zen 2 architecture detected. BMI2 instructions may not be performant."
fi

rm arch.txt
