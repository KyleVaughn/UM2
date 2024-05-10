# Description: This script computes the predicted spectral radius of the CMFD 
# accelerated fixed-source iteration using Fourier analysis. The script is
# for a 1D, 1-group problem based upon the following paper:
#   Analysis of Stabilization Techniques for CMFD
#   Acceleration of Neutron Transport Problems
#   by M. Jarrett, B. Kochunas, A. Zhu, and T. Downar
#   Nuclear Science and Engineering 2016

import numpy as np
import matplotlib.pyplot as plt

#===============================================================================
# Inputs
#===============================================================================
p = 4 # Fine cells per coarse cell
h = 1 # Fine cell thickness (cm)
N = 8 # Polar quadrature order, angle in (0, 2pi)
sigma_t = 1 # Total cross section (cm^-1)
c = 0.95 # Scattering ratio
s = 1 # Number of sweeps
eta = 0 # Diffusion coefficient modifier

# Computed parameters
delta = p * sigma_t * h # Equation 56
D = 1 / (3 * sigma_t) # Diffusion coefficient (cm)

#===============================================================================
# Input checking
#===============================================================================
if p < 1:
    raise ValueError('p must be at least 1')

if h <= 0:
    raise ValueError('h must be positive')

if N % 4 != 0:
    raise ValueError('N must be a multiple of 4')

if sigma_t <= 0:
    raise ValueError('sigma_t must be positive')

if c < 0 or c > 1:
    raise ValueError('c must be between 0 and 1')

if s < 1:
    raise ValueError('s must be at least 1')

if eta < 0 or eta > 0.25:
    raise ValueError('eta must be between 0 and 0.25')

#===============================================================================
# Variable naming
#===============================================================================
# j = fine cell index
# l = outer source iteration index
# s = inner source iteration index
# n = quadrature angle index 

#===============================================================================
# Helper functions
#===============================================================================

def get_chebyshev_quadrature():
    angle_begin = np.pi / N
    angle_end = 2 * np.pi - angle_begin
    polar_angles = np.linspace(angle_begin, angle_end, N)
    w = np.zeros(N) + 1 / N # Polar quadrature weights
    return [w, polar_angles]

# Equation 5
def compute_beta(mu):
    # h is a constant, so we don't need to pass it as an argument
    # Beta = (1 + exp(-h / mu)) / (1 - exp(-h / mu)) - 2 * mu / h
    # Note: coth(x) = (1 + exp(-2x)) / (1 - exp(-2x))
    # Beta = coth(h / (2 * mu)) - 2 * mu / h
    # Let x = h / (2 * mu)
    # Beta = coth(x) - 1 / x
    x = h / (2 * mu) 
    return 1 / np.tanh(x) - 1 / x

# Equation 51
def make_A_matrix(A, mu_n, alpha):
    # Assume A is zero everywhere but the locations we will fill in
    # Ensure the matrix is p by p
    if A.shape[0] != p or A.shape[1] != p:
        raise ValueError('Matrix A must be p by p')
    # Let x = (1 - beta) / 2
    # Let y = (1 + beta) / 2
    # We will fill the diagonal with x and the upper off-diagonal with y
    # Lastly, the lower left corner of the matrix with y * exp(i * alpha * delta)
    beta = compute_beta(mu_n)
    x = (1 - beta) / 2
    y = (1 + beta) / 2
    # Assume the matrix is already zeroed out
    for i in range(p):
        A[i, i] = x
        if i < p - 1:
            A[i, i + 1] = y
    A[p - 1, 0] = y * np.exp(1j * alpha * delta)

# Equation 52
def make_B_matrix(B, A, mu_n, alpha):
    # Assume B is zero everywhere but the locations we will fill in
    # Ensure the matrix is p by p
    if A.shape[0] != p or A.shape[1] != p:
        raise ValueError('Matrix A must be p by p')
    if B.shape[0] != p or B.shape[1] != p:
        raise ValueError('Matrix B must be p by p')
    # Let x = mu_n / (sigma_t * h)
    # We will fill the diagonal with -x and the upper off-diagonal with x 
    # Lastly, the lower left corner of the matrix with x * exp(i * alpha * delta)
    x = mu_n / (sigma_t * h) 
    # Assume the matrix is already zeroed out
    for i in range(p):
        B[i, i] = -x
        if i < p - 1:
            B[i, i + 1] = x
    B[p - 1, 0] = x * np.exp(1j * alpha * delta)
    B += A

# Equation 54 
def make_U_matrix(U, A, B, B_inv, alpha):
    # Ensure the matrix is p by p
    if A.shape[0] != p or A.shape[1] != p:
        raise ValueError('Matrix A must be p by p')
    if B.shape[0] != p or B.shape[1] != p:
        raise ValueError('Matrix B must be p by p')
    if B_inv.shape[0] != p or B_inv.shape[1] != p:
        raise ValueError('Matrix B_inv must be p by p')
    if U.shape[0] != p or U.shape[1] != p:
        raise ValueError('Matrix U must be p by p')

    # Zero out U
    U *= 0.0

    # U = c / 2 sum_n^N w_n A_n (B_n)^-1
    for n in range(N):
        make_A_matrix(A, mu[n], alpha)
        print(A)
        make_B_matrix(B, A, mu[n], alpha)
        B_inv = np.linalg.inv(B)
        U += w[n] * (A @ B_inv) 
    U *= c / 2


#===============================================================================
# Set up polar angle quadrature
#===============================================================================
[w, polar_angles] = get_chebyshev_quadrature() 
mu = np.cos(polar_angles) # Cosine of polar angles

#===============================================================================
# Set up linear system 
#===============================================================================

alpha = np.pi / (2 * delta)
print('alpha =', alpha)

A = np.zeros((p, p), dtype=complex)
B = np.zeros((p, p), dtype=complex)
B_inv = np.zeros((p, p), dtype=complex)
U = np.zeros((p, p), dtype=complex)
J = np.ones((p, p), dtype=complex) # Does not change
omega = np.zeros((p, p), dtype=complex)

make_U_matrix(U, A, B, B_inv, alpha)
print(U)

# Visualize the matrix
plt.matshow(np.abs(U))
plt.colorbar()
plt.show()

