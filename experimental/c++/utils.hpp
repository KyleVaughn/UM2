#pragma once

#include <iostream>
#include <vector>
#include <benchmark/benchmark.h>

const size_t LINE_PADDING = 64;

void format_string(std::string& s)
{
    s.insert(s.begin(), LINE_PADDING - s.length(), ' ');
}
