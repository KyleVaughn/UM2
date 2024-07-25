import matplotlib.pyplot as plt
import numpy as np
import sys

# Get the file name from the command line
if len(sys.argv) != 2:
    print('Usage: python plot_spectrum.py <file_name>')
    sys.exit(1)

file_name = sys.argv[1]

# Read file with data of the form: group, neutrons/sec
data = np.loadtxt(file_name, delimiter=',')
neutrons = data[:, 1]

total_neutrons = np.sum(neutrons)
plt.stairs(neutrons/total_neutrons)
plt.xlabel('Group')
plt.ylabel('Normalized Neutrons/sec')
plt.title('Normalized Energy Spectrum')
plt.show()
