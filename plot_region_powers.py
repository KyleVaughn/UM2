import matplotlib.pyplot as plt
import numpy as np
import sys

# Get the file name from the command line
if len(sys.argv) != 2:
    print('Usage: python plot_region_powers.py <file_name>')
    sys.exit(1)

file_name = sys.argv[1]

# Read file with data of the form: power, x, y, z
# skiprows=1 to skip the header
data = np.loadtxt(file_name, delimiter=',', skiprows=1)
powers = data[:, 0]
x = data[:, 1]
y = data[:, 2]

plt.scatter(x, y, c=powers, cmap='viridis')
plt.colorbar(label='Power')
plt.show()
