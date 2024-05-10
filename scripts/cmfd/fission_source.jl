#--------------------------------------------------------------------------
# Description
#--------------------------------------------------------------------------
# This script computes the predicted spectral radius of the CMFD
# accelerated fission-source iteration using Fourier analysis. The script is
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
# The cell has thickness 1 cm, wich makes Σ_t == Δ

struct Params
    p::Int          # Fine cells per coarse cell
    N::Int          # Polar quadrature order, θ ∈ (0, 2π)
    Σ_t::Float64    # Total cross section (cm^-1)
    c::Float64      # Scattering ratio
    σ::Int          # Number of sweeps
    η::Float64      # Diffusion coefficient modifier η ∈ [0, 1/4]
    num_α::Int      # Number of equally spaced points to sample α ∈ [0, π/Δ]
    # Computed parameters
    h::Float64      # Fine cell thickness (cm)
    Δ::Float64      # Equation 56, optical thickness of the coarse cell
    D::Float64      # Equation 27

    Params(p, N, Σ_t, c, σ, η, num_α) = 
        new(p, N, Σ_t, c, σ, η, num_α, 
            1.0 / p, # Domain fixed at 1 cm 
            Σ_t, # p * Σ_t * h = Σ_t
            1 / (3 * Σ_t) + η)
end

params = Params(
    4, # p 
    32, # N
    0.1, # Σ_t
    0.95, # c
    1, # σ
    0.0, # η
    3 # num_α
)

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
    # β(μ) = (1 + exp(-h / μ)) / (1 - exp(-h / μ)) - 2μ / h
    # Note: coth(x) = (1 + exp(-2x)) / (1 - exp(-2x))
    # Hence, β(μ) = coth(h / 2μ) - 2μ / h
    # Let x = h / 2μ
    # Beta = coth(x) - 1 / x
    x = h / 2μ
    return coth(x) - inv(x)
end

# Equation 51
# No allocations
function setAₙ!(Aₙ, params, μₙ, α)
    # Assume Aₙ is zero everywhere except the locations we will set.
    # This way we can reuse the matrix for each μₙ
    # Let x = (1 - βₙ) / 2
    # Let y = (1 + βₙ) / 2
    # We will fill the diagonal with x and the upper off-diagonal with y
    # Lastly, the lower left corner of the matrix with y * exp(i α Δ)
    βₙ = β(params.h, μₙ)
    x = (1 - βₙ) / 2
    y = (1 + βₙ) / 2
    Aₙ[1, 1] = x
    p = params.p
    for i in 2:p
        Aₙ[i - 1, i] = y
        Aₙ[i, i] = x
    end
    Aₙ[p, 1] = y * exp(im * α * params.Δ)
    return Aₙ
end

# Equation 52
function setBₙ!(Bₙ, Aₙ, params, μₙ, α)
    # Assume B is zero everywhere except the locations we will set.
    # Let x = μₙ / (Σ_t * h)
    # We will fill the diagonal with -x and the upper off-diagonal with x
    # Lastly, the lower left corner of the matrix with x * exp(i * α * Δ)
    x = μₙ / (params.Σ_t * params.h)
    Bₙ[1, 1] = -x
    p = params.p
    for i in 2:p
        Bₙ[i - 1, i] = x
        Bₙ[i, i] = -x
    end
    Bₙ[p, 1] = x * exp(im * α * params.Δ)
    Bₙ .+= Aₙ
    return Bₙ
end

# Equation 54
function setU!(U, Aₙ, Bₙ, params, w, μ, α)
    # Zero out U
    U .= zero(Complex{Float64})

    # Sadly we allocate memory for X = Bₙ \ Aₙ each loop :(
    # U = (c / 2) ∑_n^N wₙ Aₙ (Bₙ)^-1
    for n in 1:params.N
        setAₙ!(Aₙ, params, μ[n], α) 
        setBₙ!(Bₙ, Aₙ, params, μ[n], α) 
        # We wish to find a matrix X such that Aₙ * (Bₙ)^-1 = X 
        # Or Aₙ = X * Bₙ
        # We may use the '\' operator to find X = Bₙ \ Aₙ
        X = Bₙ \ Aₙ
        @. U += w[n] * X
    end
    U .*= params.c / 2
end

# Equation 56
function getF(params, α)
    c = params.c
    Δ = params.Δ
    num = c * params.Σ_t * params.h
    den_l = 2 * params.D * (1 - cos(α * Δ))
    den_r = (1 - c) * Δ
    return num / (den_l + den_r) 
end  

# Equation 57
function setω!(ω, Aₙ, Bₙ, U, params, w, μ, α)
    setU!(U, Aₙ, Bₙ, params, w, μ, α)
    F = getF(params, α)
    J = ones(params.p, params.p)
    σ = params.σ
    println("U^σ = ", U^σ)
    ω .= (U^σ + F * J * (U^σ - U^(σ - 1)))
end

function ρ(params::Params)
    # Set up polar angle quadrature
    (w, polar_angles) = getChebyshevQuadrature(params.N)
    μ = cos.(polar_angles) # Cosine of polar angles

    # Set up α samples
    α = LinRange(0, π / params.Δ, params.num_α) 
    largest_eigvals = zeros(params.num_α)

    # Allocate memory for matrices
    Aₙ = zeros(Complex{Float64}, params.p, params.p)
    Bₙ = zeros(Complex{Float64}, params.p, params.p)
    U = zeros(Complex{Float64}, params.p, params.p)
    ω = zeros(Complex{Float64}, params.p, params.p)
    # For each α, compute the spectral radius of the ω matrix
    for (i, α_i) in enumerate(α)
        setω!(ω, Aₙ, Bₙ, U, params, w, μ, α_i)
        eigens = eigvals(ω)
        println("α = ", α_i)
        for eig in eigens
            println("eig = ", eig)
        end
        largest_eigvals[i] = maximum(abs.(eigens)) 
        println("α = ", α_i, " λ = ", largest_eigvals[i])
    end
    return maximum(largest_eigvals) 
end

ρ(params)
