START_DIR=$PWD
cd ../
# Has to be followed by "justified", "justification", etc.
grep -r "NOLINTBEGIN" | grep -v "justif"
grep -r "NOLINTNEXTLINE" | grep -v "justif"
grep -r "cppcheck-suppress" | grep -v "justif"
cd $START_DIR
