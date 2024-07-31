import matplotlib.pyplot as plt
import numpy as np
import sys

# Get the file name from the command line
if len(sys.argv) != 2:
    print('Usage: python plot_xsecs.py <file_name>')
    sys.exit(1)

file_name = sys.argv[1]

names = []
a = []
f = []
nuf = []
tr = []
s = []

with open(file_name) as file:
    # Get the number of materials and number of groups
    num_materials, num_groups = map(int, file.readline().split())
    for i in range(num_materials):
        # Get the name of the material
        names.append(file.readline().strip())
        # For each group, read:
        # a f nuf tr s
        this_a = np.zeros(num_groups) 
        this_f = np.zeros(num_groups)
        this_nuf = np.zeros(num_groups)
        this_tr = np.zeros(num_groups)
        this_s = np.zeros(num_groups)
        for g in range(num_groups):
            this_a[g], this_f[g], this_nuf[g], this_tr[g], this_s[g] = \
                    map(float, file.readline().split())
        a.append(this_a)
        f.append(this_f)
        nuf.append(this_nuf)
        tr.append(this_tr)
        s.append(this_s)

# Plot the cross sections
# Absorption
plt.figure()
for i in range(num_materials):
    plt.stairs(a[i], label=names[i])
plt.xlabel('Group')
plt.ylabel('Absorption Cross Section (cm$^{-1}$)')
plt.yscale('log')
plt.title('Absorption Cross Sections')
plt.legend()
plt.show()

# Fission
plt.figure()
for i in range(num_materials):
    plt.stairs(f[i], label=names[i])
plt.xlabel('Group')
plt.ylabel('Fission Cross Section (cm$^{-1}$)')
plt.yscale('log')
plt.title('Fission Cross Sections')
plt.legend()
plt.show()

# NuFission
plt.figure()
for i in range(num_materials):
    plt.stairs(nuf[i], label=names[i])
plt.xlabel('Group')
plt.ylabel('NuFission Cross Section (cm$^{-1}$)')
plt.yscale('log')
plt.title('NuFission Cross Sections')
plt.legend()
plt.show()

# Transport
plt.figure()
for i in range(num_materials):
    plt.stairs(tr[i], label=names[i])
plt.xlabel('Group')
plt.ylabel('Transport Cross Section (cm$^{-1}$)')
plt.yscale('log')
plt.title('Transport Cross Sections')
plt.legend()
plt.show()

# Scattering
plt.figure()
for i in range(num_materials):
    plt.stairs(s[i], label=names[i])
plt.xlabel('Group')
plt.ylabel('Scattering Cross Section (cm$^{-1}$)')
plt.yscale('log')
plt.title('Scattering Cross Sections')
plt.legend()
plt.show()

# Total
plt.figure()
for i in range(num_materials):
    plt.stairs(a[i] + s[i], label=names[i])
plt.xlabel('Group')
plt.ylabel('Total Cross Section (cm$^{-1}$)')
plt.yscale('log')
plt.title('Total Cross Sections')
plt.legend()
plt.show()
