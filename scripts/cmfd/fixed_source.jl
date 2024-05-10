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

using BenchmarkTools
using Plots
using FastGaussQuadrature
using LinearAlgebra

const FloatType = Float64

# ASSUMPTIONS: 
#   - K = 1: A single coarse cell with periodic boundary conditions.
#   - The cell has thickness 1 cm, which makes Δ = p * Σ_t * h = Σ_t

struct Params
    p::Int          # Fine cells per coarse cell
    N::Int          # Polar quadrature order, θ ∈ (0, 2π), hence μ ∈ (-1, 1)
    Σ_t::FloatType    # Total cross section (cm^-1)
    c::FloatType      # Scattering ratio
    σ::Int          # Number of sweeps
    η::FloatType      # Diffusion coefficient modifier η ∈ [0, 1/4]
    num_α::Int      # Number of equally spaced points to sample α ∈ [0, π/Δ]
    # Computed parameters
    h::FloatType      # Fine cell thickness (cm)
    Δ::FloatType      # Equation 56, optical thickness of the coarse cell
    D::FloatType      # Equation 27

    function Params(p, N, Σ_t, c, σ, η, num_α)
        @assert 1 <= p "p must be greater than or equal to 1"
        @assert N % 2 == 0 "N must be a multiple of 2" 
        @assert 0 < Σ_t "Σ_t must be greater than 0"
        @assert 0 <= c <= 1 "c must be between 0 and 1" 
        @assert 1 <= σ "σ must be greater than or equal to 1"
        @assert 0 <= η <= 1 / 4 "η must be between 0 and 1/4"
        @assert 1 <= num_α "num_α must be greater than or equal to 1"

        return new(p, N, Σ_t, c, σ, η, num_α, 
            1.0 / p, # Domain fixed at 1 cm 
            Σ_t, # p * Σ_t * h = Σ_t
            1 / (3 * Σ_t) + η
           )
    end
end

#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------

# Equation 5
# No allocation of memory is done in this function
function βₙ(h, μₙ)
    # βₙ(h, μₙ) = \frac{
    #               1 + e^{-h / μₙ}
    #             }{
    #               1 - e^{-h / μₙ}
    #             } 
    #             - \frac{2μₙ}{h}
    #
    # Let x = -h / 2μₙ. Substituting x into the equation above we get
    # βₙ(x) = \frac{
    #           1 + e^{2x}
    #         }{
    #           1 - e^{2x}
    #         } 
    #         + \frac{1}{x}
    # Note: 
    #   coth(x) = \frac{
    #               e^{2x} + 1
    #             }{
    #               e^{2x} - 1
    #             }
    #
    # Hence, β(x) = -coth(x) + 1 / x
    # Noting that coth(x) = -coth(-x) we can simplify the equation
    # using z = -x
    z = h / 2μₙ
    return coth(z) - 1 / z
end

# Sanity check on βₙ algebra
function test_βₙ()
    h = 0.25
    μₙ = 0.339981
    computed = βₙ(h, μₙ)
    expected = (1 + exp(-h / μₙ)) / (1 - exp(-h / μₙ)) - 2μₙ / h
    @assert computed ≈ expected "βₙ failed"
end
test_βₙ()

# Equation 51
# No allocation of memory is done in this function
function setAₙ!(Aₙ, params, μₙ, α)
    # ASSUMPTIONS: 
    #   - Aₙ is zero everywhere except the locations we will set
    # 
    # Using the assumption above we can reuse the matrix for each μₙ
    
    # Let d = (1 - βₙ) / 2 be the value on the diagonal
    # Let u = (1 + βₙ) / 2 be the value on the upper diagonal
    # The lower left corner of the matrix is u * exp(i α Δ)
    β = βₙ(params.h, μₙ)
    d = (1 - β) / 2
    u = (1 + β) / 2

    # Traverse in column-major order
    p = params.p
    Aₙ[1, 1] = d
    Aₙ[p, 1] = u * exp(im * α * params.Δ)
    for i in 2:p
        Aₙ[i - 1, i] = u
        Aₙ[i, i] = d
    end
    return Aₙ
end

# Equation 52
# No allocation of memory is done in this function
function setBₙ!(Bₙ, Aₙ, params, μₙ, α)
    # ASSUMPTIONS:
    #  - Bₙ is zero everywhere except the locations we will set
    #  - Aₙ is already set

    # Let d = -μₙ / (Σ_t * h) be the value on the diagonal
    # Let u = μₙ / (Σ_t * h) be the value on the upper diagonal
    # The lower left corner of the matrix is u * exp(i α Δ)
    # Note: d = -u 
    u = μₙ / (params.Σ_t * params.h)
    d = -u
    
    # Traverse in column-major order
    p = params.p
    Bₙ[1, 1] = d
    Bₙ[p, 1] = u * exp(im * α * params.Δ)
    for i in 2:p
        Bₙ[i - 1, i] = u
        Bₙ[i, i] = d
    end
    Bₙ .+= Aₙ
    return Bₙ
end

# Equation 54
# Allocates memory for X = Bₙ \ Aₙ for each N (at least N  allocations of p^2)
function setU!(U, Aₙ, Bₙ, params, w, μ, α)
    # ASSUMPTIONS:
    #  - Aₙ and Bₙ are zero everywhere except the locations we will set

    # Zero out U
    U .= zero(Complex{FloatType})

    # Loop over the polar quadrature to compute equation 54
    # U = (c / 2) ∑_n^N wₙ * Aₙ * (Bₙ)^-1
    # However, instead of computing (Bₙ)^-1 we will compute X = Bₙ \ Aₙ,
    # which is a matrix such that Aₙ = X * Bₙ. Equivalently, Aₙ * (Bₙ)^-1 = X 
    # Sadly we allocate memory for X = Bₙ \ Aₙ each loop :(
    N = params.N
    for n in 1:N
        setAₙ!(Aₙ, params, μ[n], α) 
        setBₙ!(Bₙ, Aₙ, params, μ[n], α) 
        X = Bₙ \ Aₙ
        @. U += w[n] * X
    end
    U .*= params.c / 2
    return U
end

# Equation 56
# No allocation of memory is done in this function
function getF(params, α)
    c = params.c
    Δ = params.Δ
    num = c * params.Σ_t * params.h
    den_l = 2 * params.D * (1 - cos(α * Δ))
    den_r = (1 - c) * Δ
    return num / (den_l + den_r) 
end  

# Equation 57 (Just constructing the matrix)
# Allocates:
#   - N times from setU! (N times p^2)
#   - Matrix temporaries for ω .= (U^σ + F * J * (U^σ - U^(σ - 1)))
function setω!(ω, Aₙ, Bₙ, U, J, params, w, μ, α)
    # ASSUMPTIONS:
    #  - Aₙ and Bₙ are zero everywhere except the locations we will set
    #  - J is a matrix of ones

    setU!(U, Aₙ, Bₙ, params, w, μ, α)
    F = getF(params, α)
    σ = params.σ
    # Benchmarking shows that even if you get clever with factoring out
    # U^(σ - 1), the timing is about the same.
    ω .= (U^σ + F * J * (U^σ - U^(σ - 1)))
    return ω
end

function ρ(params::Params)
    println("Computing spectral radius for parameters: ", params)

    # Set up polar angle quadrature
    μ, w = gausslegendre(params.N)

    # Sanity check on quadrature
    @assert sum(w) ≈ 2 "Polar quadrature weights do not sum to 2"
    @assert -1 < minimum(μ) "Polar quadrature minimum value is less than -1"
    @assert maximum(μ) < 1 "Polar quadrature maximum value is greater than 1"

    # Set up α samples. We sample α ∈ [0, π/Δ] (Equation 58)
    α = collect(LinRange(0, π / params.Δ, params.num_α)) 
    largest_eigvals = zeros(params.num_α)

    # Allocate memory for matrices
    Aₙ = zeros(Complex{FloatType}, params.p, params.p)
    Bₙ = zeros(Complex{FloatType}, params.p, params.p)
    U = zeros(Complex{FloatType}, params.p, params.p)
    J = ones(Complex{FloatType}, params.p, params.p)
    ω = zeros(Complex{FloatType}, params.p, params.p)
    # For each α, compute the spectral radius of the ω matrix
    for (i, α_i) in enumerate(α)
        setω!(ω, Aₙ, Bₙ, U, J, params, w, μ, α_i)
        eigens = eigvals(ω) # Allocates vector of size p
        largest_eigvals[i] = maximum(abs.(eigens)) # Allocates vector of size p
    end
    return maximum(largest_eigvals) 
end

#--------------------------------------------------------------------------
# Parametric study
#--------------------------------------------------------------------------
# Use the same quadrature order and number of α samples for all cases 
const N = 32
const num_α = 1000

# Figure 3
# Scattering ratio vs spectral radius for varying Δ (Σ_t)
c = collect(LinRange(0.001, 0.999, 15))
Δ = [0.001, 0.4, 0.8, 1.6, 2, 2.5, 4]
p = 4
σ = 1
η = 0.0
r = zeros(length(c), length(Δ))

plot()
for (i, Δ_i) in enumerate(Δ)
    for (j, c_j) in enumerate(c)
        params = Params(p, N, Δ_i, c_j, σ, η, num_α)
        r[j, i] = ρ(params)
    end
    plot!(c, r[:, i], label="Δ = $Δ_i")
end
xlabel!("Scattering ratio")
ylabel!("Spectral radius")
title!("p = $p, σ = $σ, η = $η")
# savefig("scattering_ratio_vs_spectral_radius.png")
display(plot!())

# Plot as a scatter plot
#scatter()
#for (i, Δ_i) in enumerate(Δ)
#    Δ_i_vec = fill(Δ_i, length(c))
#    println(Δ_i_vec)
#    scatter!(fill(Δ_i, length(c)), c, zcolor=r[:, i]) 
#end
#display(scatter!())

heatmap(r)








