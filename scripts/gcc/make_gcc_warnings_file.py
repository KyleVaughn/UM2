import os
all_warnings_file = 'all_gcc_warnings.txt'
cmd = 'g++ --help=warnings > ' + all_warnings_file
print(cmd)
os.system(cmd)
with open(all_warnings_file) as f:
    all_warnings = f.readlines() 

# Only get the first word of each line, if it's not empty
all_warnings = [line.split()[0] for line in all_warnings if line.split()]

# Append a new line to the end of each line
all_warnings = [line + '\n' for line in all_warnings]

default_flags = [
        '-Wall',
        '-Wextra',
        '-pedantic',
        '-Wconversion',
        '-Wshadow',
        '-Wnon-virtual-dtor',
        '-Wold-style-cast',
        '-Wcast-align',
        ]
flags_str = ' '.join(default_flags)
test_cpp_file = '../../src/common/color.cpp'
default_warnings_file = 'um2_default_gcc_warnings.txt'
cmd = 'g++ -Q ' + flags_str + " " +  test_cpp_file + \
        ' --help=warnings | grep enabled > ' + default_warnings_file
print(cmd)
os.system(cmd)
with open(default_warnings_file) as f:
    # Only get the first word of each line
    default_warnings = [line.split()[0] for line in f.readlines()]

# Remove the default warnings from the list of all warnings
for line in default_warnings:
    if line in all_warnings:
        print(line)
        all_warnings.remove(line)
## Remove any warnings containing:
##   "-Wc++", "-Wc99", "-Wc2x", and "-Wpre-" since we use C++23
##   "-Wthread" since we don't use multithreading yet
##   "remark_" since we don't care about remarks
##   "warn_doc_" and "-Wdocumentation" since we don't care about documentation warnings
#prefixes_to_remove = ['-Wc++', '-Wc99', '-Wc2x', '-Wpre-', '-Wthread', 'remark_', 
#        'warn_doc_', '-Wdocumentation']
#for prefix in prefixes_to_remove:
#    all_warnings = [line for line in all_warnings if prefix not in line]
print(len(all_warnings))

# Write the new file
with open('gcc_warnings.txt', 'w') as f:
    f.writelines(all_warnings)
