# Find the difference between the set of all clang warnings and the set of
# warnings enabled by a "current" set of clang warnings, minus a set of
# warnings we don't care about, like C++98 warnings.

import subprocess 

diagtool = 'diagtool-15'
all_warnings_file = 'all_warnings.txt'
test_cpp_file = 'point2d.cpp' # Any file will do
current_warnings_file = 'current_warnings.txt'
current_flags = [
        '-Wall', 
        '-Wextra', 
        '-pedantic', 
        '-Wconversion', 
        '-Wdouble-promotion',
        '-Wcast-qual',
        '-Wconditional-uninitialized',
        '-Wshadow-all', 
        '-Wfloat-equal',
        '-Wnon-virtual-dtor', 
        '-Wold-style-cast', 
        '-Wcast-align',
        '-Woverriding-method-mismatch',
        '-Wheader-hygiene',
        '-Wimplicit-fallthrough',
        '-Wunreachable-code-aggressive',
        '-Wmissing-noreturn',
        '-Wrange-loop-analysis',
        '-Wreserved-identifier',
        '-Wshift-sign-overflow',
        '-Wtautological-type-limit-compare',
        '-Wtautological-unsigned-char-zero-compare',
        '-Wtautological-unsigned-enum-zero-compare',
        '-Wtautological-unsigned-zero-compare',
        '-Wtautological-value-range-compare',
        '-Wthread-safety',
        '-Wunused',
        '-Wunaligned-access',
        '-Wundef',
        '-Werror',
        '-Wunneeded-member-function',
        '-Wvector-conversion',
        '-Wvla',
        '-Wzero-as-null-pointer-constant',
        ]
current_flags.sort()
remove_flags = [
        '-Wc++*',
        '-Wpre-c++1*',
        '-Wpre-c++20*',         # we use C++20
        '-Wdocumentation*',     # we don't care about documentation warnings
        '-Wobjc*',              # we don't use Objective-C
        '-Wmodule*',            # we don't use modules
        '-Wnon-modular*',       # we don't use modules
        '-Wunused*',            # unused warnings may give false positives due to templates
        ]
diagtool_cmd = diagtool + ' list-warnings > ' + all_warnings_file
p = subprocess.call(diagtool_cmd, shell=True)
if p != 0:
    print('Error running diagtool')
    exit(p)
with open(all_warnings_file) as f:
    all_warnings = f.readlines()
all_flags = []
for line in all_warnings:
    open_bracket = line.find('[')
    close_bracket = line.find(']')
    if open_bracket == -1 or close_bracket == -1:
        continue
    flag = line[open_bracket+1:close_bracket]
    flag = flag.strip()
    all_flags.append(flag)
all_flags = list(set(all_flags))
all_flags.sort()
print("There are " + str(len(all_flags)) + " unique flags")

flags_str = ' '.join(current_flags)
diagtool_cmd = diagtool + ' show-enabled ' + flags_str + ' ' + \
    test_cpp_file + ' > ' + current_warnings_file 
p = subprocess.call(diagtool_cmd, shell=True)
if p != 0:
    print('Error running diagtool')
    exit(p)
with open(current_warnings_file) as f:
    # Remove the first character of each line (W, E, etc.)
    default_warnings = [line[1:] for line in f.readlines()]

# The lower level flags enabled by the current flags
current_lower_flags = []
for line in default_warnings:
    open_bracket = line.find('[')
    close_bracket = line.find(']')
    if open_bracket == -1 or close_bracket == -1:
        continue
    flag = line[open_bracket+1:close_bracket]
    flag = flag.strip()
    current_lower_flags.append(flag)
current_lower_flags = list(set(current_lower_flags))
print("There are " + str(len(current_lower_flags)) + \
        " flags enabled by the current set of flags")

diff_flags = []
for flag in all_flags:
    if flag not in current_lower_flags:
        diff_flags.append(flag)
diff_flags.sort()
print("There are " + str(len(diff_flags)) + " flags unaccounted for")

for flag in remove_flags:
    if flag[-1] == '*':
        diff_flags = [f for f in diff_flags if not f.startswith(flag[:-1])]

print("There are " + str(len(diff_flags)) + " flags unaccounted for after removing flags we don't care about")
for flag in diff_flags:
    print(flag)
