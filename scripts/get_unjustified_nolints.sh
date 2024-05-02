#!/bin/bash

START_DIR=$PWD
cd ../

# Has to be followed by "OK" or "match"
grep -r "NOLINTBEGIN" | grep -v "OK" | grep -v "match"
grep -r "NOLINTNEXTLINE" | grep -v "OK" | grep -v "match"

cd $START_DIR
