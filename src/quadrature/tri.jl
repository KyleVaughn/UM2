# Dunavant, D. (1985). High degree efficient symmetrical Gaussian quadrature rules
# for the triangle. International journal for numerical methods in engineering, 21(6),
# 1129-1148.
using LinearAlgebra
using Optim
using StaticArrays
using BenchmarkTools

setprecision(BigFloat, 512)
eps_256 = 1e-78
const T = BigFloat 

# Polynomial degree
# const p = parse(Int64, ARGS[1])
const p = 6

# m values from Table I.
const m_all = SVector(1, 2, 3, 4, 5, 7, 8, 10, 12, 14, 16, 19, 21, 24, 27, 30, 33, 37, 40, 44)
const m = m_all[p]

# Polar moments from Table I.
const ν_all = SVector( 
        +1//1,
        +1//4,
        -1//10,
        +1//10,
        -2//35,
       +29//560,
        +1//28,
        -1//28,
       +11//350,
        +1//40,
       -37//1540,
        -1//55,
       +13//616,
        +1//55,
       -49//2860,
        -2//143,
      +425//28028,
      +137//10010,
        +1//91,
       -64//5005,
        -1//91,
      +523//45760,
       +85//8008,
        +1//112,
     -6733//680680,
      -109//12376,
        -1//136,
      +217//24310,
      +209//24752,
        +1//136,
     -2909//369512,
       -65//9044,
        -2//323,
    +66197//9237800,
     +8069//1175720,
      +317//51680,
        +1//190,
     -3769//587860,
       -77//12920,
        -1//190,
    +83651//14226212,
    +11303//1989680,
       +92//17765,
        +1//220)
const ν = SVector{m, T}(ν_all[1:m])

# Polar moment indices from Table I.
const iν_all = SMatrix{44, 2, Int64, 88}(
    0, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 
    0, 0, 3, 0, 3, 0, 6, 3, 0, 6, 3, 9,  0,  6,  3,  9,  0,  6, 12,  3,  9,  0,  6, 12,  3,  9, 15,  0,  6, 12,  3,  9, 15,  0,  6, 12, 18,  3,  9, 15,  0,  6, 12, 18
   )
const iν = SMatrix{m, 2, Int64, 2m}(iν_all[1:m, :])

# n₀, n₁, n₂ values provided in Table IV, 
# the number of groups of multiplicity 1, 3, and 6
const nᵢ= SVector((1, 0, 0),
                  (0, 1, 0),
                  (1, 1, 0),
                  (0, 2, 0),
                  (1, 2, 0),
                  (0, 2, 1),
                  (1, 2, 1),
                  (1, 3, 1),
                  (1, 4, 1),
                  (1, 2, 3),
                  (0, 5, 2),
                  (0, 5, 3),
                  (1, 6, 3),
                  (0, 6, 4),
                  (0, 6, 5),
                  (1, 7, 5),
                  (1, 8, 6),
                  (1, 9, 7),
                  (1, 8, 8),
                  (1, 10, 8))
const n₀, n₁, n₂ = nᵢ[p]
# calculate the number of unknowns
const n = n₀ + 2n₁ + 3n₂
# calculate the number of gaussian points and weights
const ng = n₀ + 3n₁ + 6n₂

# Given α, β, and weights in Appendix II
if p === 1
    const αβw_Float64 = @SVector [
      (0.333333333333333, 0.333333333333333, 0.500000000000000)]
elseif p === 2
    const αβw_Float64 = @SVector [
      (0.666666666666667, 0.166666666666667, 0.166666666666667)]
elseif p === 3
    const αβw_Float64 = @SVector [
      (0.333333333333333, 0.333333333333333, -0.281250000000000),
      (0.600000000000000, 0.200000000000000, 0.260416666666667)]
elseif p === 4
    const αβw_Float64 = @SVector [
      (0.108103018168070, 0.445948490915965, 0.111690794839005),
      (0.816847572980459, 0.091576213509771, 0.054975871827661)]
elseif p === 5
    const αβw_Float64 = @SVector [
      (0.333333333333333, 0.333333333333333, 0.112500000000000),
      (0.059715871789770, 0.470142064105115, 0.066197076394253),
      (0.797426985353087, 0.101286507323456, 0.062969590272414)]
elseif p === 6
    const αβw_Float64 = @SVector [
      (0.501426509658179, 0.249286745170910, 0.058393137863189),
      (0.873821971016996, 0.063089014491502, 0.025422453185104),
      (0.053145049844816, 0.636502499121399, 0.041425537809187)]
elseif p === 7
    const αβw_Float64 = @SVector [
      (0.333333333333333, 0.333333333333333, -0.074785022233841),
      (0.479308067841920, 0.260345966079040, 0.087807628716604),
      (0.869739794195568, 0.065130102902216, 0.026673617804419),
      (0.048690315425316, 0.312865496004874, 0.038556880445128)]
elseif p === 8
    const αβw_Float64 = @SVector [
      (0.333333333333333, 0.333333333333333, 0.072157803838894),
      (0.081414823414554, 0.459292588292723, 0.047545817133643),
      (0.658861384496480, 0.170569307751760, 0.051608685267359),
      (0.898905543365938, 0.050547228317031, 0.016229248811599),
      (0.008394777409958, 0.263112829634638, 0.013615157087217)]
elseif p === 9
    const αβw_Float64 = @SVector [
      (0.3333333333330, 0.3333333333333, 0.04856789814140),
      (0.0206349616025, 0.4896825191990, 0.01566735011355),
      (0.1258208170140, 0.4370895914930, 0.03891377050240),
      (0.6235929287620, 0.1882035356190, 0.03982386946360),
      (0.9105409732110, 0.0447295133945, 0.01278883782935),
      (0.0368384120547, 0.2219629891613, 0.02164176968865)]
else
    error("Unsupported polynomial order")
end

# Given α and β, find r and a in the area function.
# x = [r, a]
function αβ_to_ra_error(x, α, β)
    r = x[1]
    a = x[2]
    α_err = (1 - 2r*cos(a))/3 - α
    β_err = (1 + r*cos(a) - sqrt(T(3))*r*sin(a))/3 - β
    return α_err^2 + β_err^2
end

function αβw_to_initial_guess()
    initial_guess = SizedVector{n, T}(zeros(T, n))
    icount = 1
    if n₀ === 1
        α,β,w = αβw_Float64[1]
        initial_guess[1] = 2w
        icount += 1
    end
    for i = n₀ + 1 : 2 : n₀ + 2n₁
        α,β,w = αβw_Float64[icount]       
        initial_guess[i]   = 6w
        initial_guess[i+1] = (1 - 3α)/2
        icount += 1
    end
    for i = n₀ + 2n₁ + 1: 3 : n
        α,β,w = αβw_Float64[icount]
        initial_guess[i] = 12w
        res = optimize(x -> αβ_to_ra_error(x, α, β),
                       [0.0, 0.0],
                       Optim.Options(g_tol=1e-30,
                                     iterations=Int64(1e4),
                                    )
                      )
        println("r, a from α, β")
        println(res)
        x = Optim.minimizer(res)
        initial_guess[i+1] = x[1]
        initial_guess[i+2] = x[2]
        icount += 1
    end
    return initial_guess
end

# The error in equation 22b for moment iM given 
# x = {w₀ if n₀ = 1} ∪ {wᵢ, rᵢ for i = 1:n₁} ∪ {wᵢ, rᵢ, 3αᵢ for i = n₁+1:n₂}
function moment_error(x::SizedVector{n, T, Vector{T}}, iM::Int64)
    err = -ν[iM]
    j = iν[iM, 1]
    k = iν[iM, 2]
    if n₀ === 1 && iM === 1
        err += x[1]
    end
    for i = n₀ + 1 : 2 : n₀ + 2n₁
        err += x[i]*x[i+1]^j
    end
    for i = n₀ + 2n₁ + 1: 3 : n
        err += x[i]*x[i+1]^j * cos(k*x[i+2])
    end
    return err 
end

# The sum or the squared error in each unknown 
function objective(x::SizedVector{n, T, Vector{T}})
    sum_err = zero(T)
    for iM = 1:m
        sum_err += moment_error(x, iM)^2
    end
    return sum_err
end

# Convert to barycentric coordinates
function area(x)
    nsum = n₀ + n₁ + n₂
    w = zeros(T, nsum) 
    α = zeros(T, nsum) 
    β = zeros(T, nsum) 
    γ = zeros(T, nsum) 
    icount = 1
    if !(n₀ === 0)
        w[icount] = x[1]
        α[icount] = T(1//3)
        β[icount] = T(1//3) 
        γ[icount] = T(1//3) 
        icount += 1 
    end
    for i = n₀ + 1 : 2 : n₀ + 2n₁ 
        w[icount] = x[i]/3
        α[icount] = (1 - 2x[i+1])/3
        β[icount] = (1 - α[icount])/2
        γ[icount] = β[icount]
        icount += 1 
    end
    for i = n₀ + 2n₁ + 1: 3 : n 
        r = x[i+1]
        a = x[i+2]
        w[icount] = x[i]/6
        α[icount] = (1 - 2r*cos(a))/3
        β[icount] = (1 + r*cos(a) - sqrt(T(3))*r*sin(a))/3
        γ[icount] = 1 - α[icount] - β[icount]
        icount += 1
    end
    return [ (w[i], α[i], β[i], γ[i]) for i ∈ 1:nsum ]
end

initial_guess = αβw_to_initial_guess() 

println("Polynomial degree: $p")
println("nᵢ = $n₀, $n₁, $n₂")    
@time res = optimize(objective, 
                     initial_guess, 
                     Optim.Options(g_tol=1e-149,
                                   iterations=Int64(1e7),
                                   time_limit = 10
                                  )
                    )
x = Optim.minimizer(res)
println(res)
wαβγ = area(x)
for t in wαβγ
    println(Float64.(t))
end
println("Errors")
for i = 1:length(wαβγ)
    α_ref,β_ref,w_ref = αβw_Float64[i]
    γ_ref = 1 - α_ref - β_ref
    w, α, β, γ = wαβγ[i]
    w_err = Float64(w)-2w_ref 
    α_err = Float64(α)-α_ref
    β_err = Float64(β)-β_ref
    γ_err = Float64(γ)-γ_ref
    errs = (w_err, α_err, β_err, γ_err)
    for err in errs
        if abs(err) < 1e-14
            print("OK ")
        else
            print(err, " ")
        end
    end
    println()
end
