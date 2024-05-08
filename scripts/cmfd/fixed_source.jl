#--------------------------------------------------------------------------
# Description 
#--------------------------------------------------------------------------
# This script computes the predicted spectral radius of the CMFD 
# accelerated fixed-source iteration using Fourier analysis. The script is
# for a 1D, 1-group problem based upon the following paper:
#   Analysis of Stabilization Techniques for CMFD
#   Acceleration of Neutron Transport Problems
#   by M. Jarrett, B. Kochunas, A. Zhu, and T. Downar
#   Nuclear Science and Engineering 2016

using Plots
using LinearAlgebra

#------------------------------------------------------------------------------
# Inputs
#------------------------------------------------------------------------------
# Assume a single coarse cell with periodic boundary conditions.
# The cell has thickness 1 cm.
p = 4     # Fine cells per coarse cell
N = 8     # Polar quadrature order, angle in (0, 2π)
Σ_t = 1.0 # Total cross section (cm^-1)
c = 0.95  # Scattering ratio
σ = 1     # Number of sweeps
η = 0.0   # Diffusion coefficient modifier

# Computed parameters
h = 1.0 / p       # Fine cell thickness (cm)
Δ = p * Σ_t * h   # Equation 56
D = 1 / (3 * Σ_t) # Diffusion coefficient (cm)

#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------

function getChebyshevQuadrature(N)
    angle_begin = π / N
    angle_end = 2 * π - angle_begin
    polar_angles = collect(LinRange(angle_begin, angle_end, N)) 
    w = zeros(N) .+ 1 / N # Polar quadrature weights
    return (w, polar_angles)
end

# Equation 5
function β(h, μ)
    # h is a constant, so we don't need to pass it as an argument
    # β(μ) = (1 + exp(-h / μ)) / (1 - exp(-h / μ)) - 2μ / h
    # Note: coth(x) = (1 + exp(-2x)) / (1 - exp(-2x))
    # Hence, β(μ) = coth(h / 2μ) - 2μ / h
    # Let x = h / 2μ
    # Beta = coth(x) - 1 / x
    x = h / 2μ 
    return coth(x) - inv(x)
end

# Equation 51
function setA!(A, h, μₙ, α, Δ)
    # Assume A is zero everywhere but the locations we will fill in
    # Ensure the matrix is p by p
    if size(A) != (p, p) 
        throw(ArgumentError("Matrix A must be p by p"))
    end
    # Let x = (1 - βₙ) / 2
    # Let y = (1 + βₙ) / 2
    # We will fill the diagonal with x and the upper off-diagonal with y
    # Lastly, the lower left corner of the matrix with y * exp(i α Δ)
    βₙ = β(h, μₙ)
    x = (1 - βₙ) / 2 
    y = (1 + βₙ) / 2
    # Assume the matrix is already zeroed out
    A[1, 1] = x
    for i in 2:p
        A[i - 1, i] = y
        A[i, i] = x
    end
    A[p, 1] = y * exp(im * α * Δ)
end

# Equation 52
function setB!(B, A, μₙ, α)
    # Assume B is zero everywhere but the locations we will fill in
    # Ensure each matrix is p by p
    if size(A) != (p, p) 
        throw(ArgumentError("Matrix A must be p by p"))
    end
    if size(B) != (p, p)
        throw(ArgumentError("Matrix B must be p by p"))
    end

    # Let x = μₙ / (Σ_t * h)
    # We will fill the diagonal with -x and the upper off-diagonal with x 
    # Lastly, the lower left corner of the matrix with x * exp(i * α * Δ)
    x = μₙ / (Σ_t * h) 
    println("x = $x")
    # Assume the matrix is already zeroed out
    B[1, 1] = -x
    for i in 2:p 
        B[i - 1, i] = x
        B[i, i] = -x
    end
    B[p, 1] = x * exp(im * α * Δ)
    B .+= A
end
#
## Equation 54 
#function setU!(U, α)
#    # Ensure the matrix is p by p
#    if size(U) != (p, p)
#        throw(ArgumentError("Matrix U must be p by p"))
#    end
#
#    Aₙ = zeros(Complex{Float64}, p, p)
#    Bₙ = zeros(Complex{Float64}, p, p)
#    Uₙ = zeros(Complex{Float64}, p, p)
#
#    # Zero out U
#    U .= zero(Complex{Float64})
#
#    # Sadly we allocate memory for B_inv in each loop :(
#    # U = c / 2 ∑_n^N wₙ Aₙ (Bₙ)^-1
#    for n in 1:N 
#        setA!(Aₙ, μ[n], α)
#        println("A = $A")
#        setB!(Bₙ, Aₙ, μ[n], α)
#        B_inv = inv(Bₙ) 
#        # Compute Uₙ = wₙ Aₙ (Bₙ)^-1
#        mul!(Uₙ, Aₙ, B_inv)
#        @. U += w[n] * Uₙ 
#    end
#    U .*= c / 2
#end
#
## Equation 57
#function setOmega!(ω, α)
#
#    
#end
#
#-----------------------------------------------------------------------------
# Set up polar angle quadrature
#-----------------------------------------------------------------------------
(w, polar_angles) = getChebyshevQuadrature(N) 
μ = cos.(polar_angles) # Cosine of polar angles

#-----------------------------------------------------------------------------
# Get the e 
#-----------------------------------------------------------------------------

ω = zeros(Complex{Float64}, p, p)

#const α = π / 2Δ 
#println("α = $α")
#
#U = zeros(Complex{Float64}, p, p)
#setU!(U, α)
#println("U = $U")
#heatmap(real.(U))
##J = np.ones((p, p), dtype=complex) # Does not change
##
##make_U_matrix(U, A, B, B_inv, alpha)
