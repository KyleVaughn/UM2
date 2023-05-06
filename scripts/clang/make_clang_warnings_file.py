import os
all_warnings_file = 'all_clang_warnings.txt'
diagtool_cmd = 'diagtool list-warnings > ' + all_warnings_file
print(diagtool_cmd)
os.system(diagtool_cmd)
with open(all_warnings_file) as f:
    all_warnings = f.readlines()

default_flags = [
        '-Wall', 
        '-Wextra', 
        '-pedantic', 
        '-Wconversion', 
        '-Wshadow', 
        '-Wnon-virtual-dtor', 
        '-Wold-style-cast', 
        '-Wcast-align',
        '-Woverriding-method-mismatch',
        ]
flags_str = ' '.join(default_flags)
test_cpp_file = '../../src/common/color.cpp'
default_warnings_file = 'um2_default_clang_warnings.txt'
diagtool_cmd = 'diagtool show-enabled ' + flags_str + ' ' + \
    test_cpp_file + ' > ' + default_warnings_file 
print(diagtool_cmd)
os.system(diagtool_cmd)
with open(default_warnings_file) as f:
    # Remove the first character of each line (W, E, etc.)
    default_warnings = [line[1:] for line in f.readlines()]

# Remove the default warnings from the list of all warnings
for line in default_warnings:
    if line in all_warnings:
        all_warnings.remove(line)

# Remove any warnings containing:
#   "-Wc++", "-Wc99", "-Wc2x", and "-Wpre-" since we use C++23
#   "-Wthread" since we don't use multithreading yet
#   "remark_" since we don't care about remarks
#   "warn_doc_" and "-Wdocumentation" since we don't care about documentation warnings
prefixes_to_remove = ['-Wc++', '-Wc99', '-Wc2x', '-Wpre-', '-Wthread', 'remark_', 
        'warn_doc_', '-Wdocumentation']
for prefix in prefixes_to_remove:
    all_warnings = [line for line in all_warnings if prefix not in line]
print(len(all_warnings))

# Write the new file
with open('clang_warnings.txt', 'w') as f:
    f.writelines(all_warnings)
