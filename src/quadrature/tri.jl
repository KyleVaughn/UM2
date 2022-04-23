# Dunavant, D. (1985). High degree efficient symmetrical Gaussian quadrature rules
# for the triangle. International journal for numerical methods in engineering, 21(6),
# 1129-1148.
using LinearAlgebra
using Optim
using StaticArrays

setprecision(BigFloat, 256)
eps_128 = 1e-78
T = Float64

# Return the n₀, n₁, and n₂ values provided in Table IV
function nvalues(p)
    if p == 1
        return (1, 0, 0)
    elseif p == 2
        return (0, 1, 0)
    elseif p == 3
        return (1, 1, 0)
    elseif p == 4
        return (0, 2, 0)
    elseif p == 5
        return (1, 2, 0)
    elseif p == 6
        return (0, 2, 1)
    elseif p == 7
        return (1, 2, 1)
    elseif p == 8
        return (1, 3, 1)
    elseif p == 9
        return (1, 4, 1)
    elseif p == 10
        return (1, 2, 3)
    elseif p == 11
        return (0, 5, 2)
    elseif p == 12
        return (0, 5, 3)
    elseif p == 13
        return (1, 6, 3)
    elseif p == 14
        return (0, 6, 4)
    elseif p == 15
        return (0, 6, 5)
    elseif p == 16
        return (1, 7, 5)
    elseif p == 17
        return (1, 8, 6)
    elseif p == 18
        return (1, 9, 7)
    elseif p == 19
        return (1, 8, 8)
    elseif p == 20
        return (1, 10, 8)
    else
        error("Only supports p ∈ [1, 20]")
    end
end

# Return the polar moments data from Table I
function nu!(ip, inu, dnu)
    @assert size(inu) == (44,2)
    @assert size(dnu) == (44,)
    icount = 1
    ippl = ip + 1
    for j = 1:ippl
        jpl = j + 1
        for k3 = 1:3:jpl
            if mod(j-1+k3-1,2) !== 1
                inu[icount, 1] = j - 1
                inu[icount, 2] = k3 - 1
                icount = icount + 1
            end
        end
    end
    dnu[ 1] =     +1//1
    dnu[ 2] =     +1//4
    dnu[ 3] =     -1//10
    dnu[ 4] =     +1//10
    dnu[ 5] =     -2//35
    dnu[ 6] =    +29//560
    dnu[ 7] =     +1//28
    dnu[ 8] =     -1//28
    dnu[ 9] =    +11//350
    dnu[10] =     +1//40
    dnu[11] =    -37//1540
    dnu[12] =     -1//55
    dnu[13] =    +13//616
    dnu[14] =     +1//55
    dnu[15] =    -49//2860
    dnu[16] =     -2//143
    dnu[17] =   +425//28028
    dnu[18] =   +137//10010
    dnu[19] =     +1//91
    dnu[20] =    -64//5005
    dnu[21] =     -1//91
    dnu[22] =   +523//45760
    dnu[23] =    +85//8008
    dnu[24] =     +1//112
    dnu[25] =  -6733//680680
    dnu[26] =   -109//12376
    dnu[27] =     -1//136
    dnu[28] =   +217//24310
    dnu[29] =   +209//24752
    dnu[30] =     +1//136
    dnu[31] =  -2909//369512
    dnu[32] =    -65//9044
    dnu[33] =     -2//323
    dnu[34] = +66197//9237800
    dnu[35] =  +8069//1175720
    dnu[36] =   +317//51680
    dnu[37] =     +1//190
    dnu[38] =  -3769//587860
    dnu[39] =    -77//12920
    dnu[40] =     -1//190
    dnu[41] = +83651//14226212
    dnu[42] = +11303//1989680
    dnu[43] =    +92//17765
    dnu[44] =     +1//220
end

function func(x,m,n,n0,n1,n2,ilb,ile,i2b,i2e,inu,dnu,T)
    f = zeros(T, m)
    for iM = 1:m
        f[iM] = -dnu[iM]
        if iM == 1 && n0 == 1
            f[1] = f[1] + x[1]
        end
        if n1 != 0 # 10
            for i = ilb:2:ile
                f[iM] = f[iM] + x[i]*x[i+1]^inu[iM,1]
            end
        end
        if n2 != 0
            for i = i2b:3:i2e
                dk3 = T(inu[iM,2])
                f[iM] = f[iM] + x[i]*x[i+1]^inu[iM,1] * cos(dk3*x[i+2])
            end
        end
    end
    return f
end

function area!(x,w,alpha,beta,gamma,n0,n1,n2,ilb,ile,i2b,i2e,inu,dnu,T)
    icount=1
    if !(n0 == 0)
        w[icount] = x[1]
        alpha[icount] = T(1//3)
        beta[icount] = (1 - alpha[icount])/2
        gamma[icount] = beta[icount]
        icount = icount + 1
    end
    if !(n1 == 0)
        for i = ilb:2:ile
            w[icount] = x[i]/3
            alpha[icount] = (1 - 2x[i+1])/3
            beta[icount] = (1 - alpha[icount])/2
            gamma[icount] = beta[icount]
            icount = icount + 1
        end
    end
    if !(n2 == 0)
        for i = i2b:3:i2e
            w[icount]=x[i]/6
            r=x[i+1]
            a=x[i+2]
            alpha[icount] = (1 - 2*r*cos(a))/3
            beta[icount] = (1 + r*cos(a) - sqrt(T(3))*r*sin(a))/3
            gamma[icount] = 1 - alpha[icount] - beta[icount]
            icount = icount + 1
        end
    end
end

initial_guesses = []
for p ∈ 1:3
    # dimension
    w     = zeros(T, 19)
    alpha = zeros(T, 19)
    beta  = zeros(T, 19)
    gamma = zeros(T, 19)
    # common
    inu   = zeros(Int64, 44, 2)
    dnu   = zeros(Rational, 44)
    # calculate the order of the polynomial
    ip = p
    # calculate the number of groups of multiplicity 1,3, and 6
    n0, n1, n2 = nvalues(ip)
    # calculate the sum of the number of groups
    nsum = n0 + n1 + n2
    # calculate the pointers for n1 and n2
    ilb = n0 + 1
    ile = ilb + 2*n1 - 1
    i2b = ile + 1
    i2e = i2b + 3*n2 - 1
    # calculate the number of gaussian points and weights
    ng = n0 + 3*n1 + 6*n2
    # calculate the number of unknowns
    n = n0 + 2*n1 + 3*n2
    # calculate the number of equations
    ipcopy = mod(ip, 6)
    ialpha = 0
    if ipcopy == 0
        ialpha = 3
    end
    if ipcopy == 1 || ipcopy == 5
        ialpha = -4
    end
    if ipcopy == 2 || ipcopy == 4
        ialpha = -1
    end
    if ipcopy == 3
        ialpha = 0
    end
    # m in Table I
    m = ((ip + 3)^2 + ialpha) ÷ 12
    nu!(ip, inu, dnu)
    f(x) = norm(func(x,m,n,n0,n1,n2,ilb,ile,i2b,i2e,inu,dnu,T))
    @time res = optimize(f, ones(T, m+3)/10, SimulatedAnnealing(), 
                         Optim.Options(f_tol=eps(T), 
                                       iterations=Int64(1e7),
                                       time_limit = 10))
    x = Optim.minimizer(res)
    push!(initial_guesses, x)
    println("Polynomial of degree ", p)
    println(res)
end
#solutions = []
#coords = []
#for p ∈ 1:10
#    # dimension
#    w     = zeros(BigFloat, 19)
#    alpha = zeros(BigFloat, 19)
#    beta  = zeros(BigFloat, 19)
#    gamma = zeros(BigFloat, 19)
#    # common
#    inu   = zeros(Int64, 44, 2)
#    dnu   = zeros(Rational, 44)
#    # calculate the order of the polynomial
#    ip = p
#    # calculate the number of groups of multiplicity 1,3, and 6
#    n0, n1, n2 = nvalues(ip)
#    # calculate the sum of the number of groups
#    nsum = n0 + n1 + n2
#    # calculate the pointers for n1 and n2
#    ilb = n0 + 1
#    ile = ilb + 2*n1 - 1
#    i2b = ile + 1
#    i2e = i2b + 3*n2 - 1
#    # calculate the number of gaussian points and weights
#    ng = n0 + 3*n1 + 6*n2
#    # calculate the number of unknowns
#    n = n0 + 2*n1 + 3*n2
#    # calculate the number of equations
#    ipcopy = mod(ip, 6)
#    ialpha = 0
#    if ipcopy == 0
#        ialpha = 3
#    end
#    if ipcopy == 1 || ipcopy == 5
#        ialpha = -4
#    end
#    if ipcopy == 2 || ipcopy == 4
#        ialpha = -1
#    end
#    if ipcopy == 3
#        ialpha = 0
#    end
#    # m in Table I
#    m = ((ip + 3)^2 + ialpha) ÷ 12
#    nu!(ip, inu, dnu)
#    f(x) = norm(func(x,m,n,n0,n1,n2,ilb,ile,i2b,i2e,inu,dnu,BigFloat))
#    @time res = optimize(f, BigFloat.(initial_guesses[ip]), 
#                         Optim.Options(g_tol=eps_128, iterations=1000000))
#    x = Optim.minimizer(res)
#    push!(initial_guesses, x)
#    area!(x,w,alpha,beta,gamma,n0,n1,n2,ilb,ile,i2b,i2e,inu,dnu,BigFloat)
#    push!(coords, (w,alpha,beta,gamma))
#
#    println("Polynomial of degree ", p)
#    println("Number of Gaussian points: ", ng)
##    println(x)
#    println(res)
##    for i = 1:nsum
##        println("   ",w[i]," ",alpha[i]," ",beta[i]," ",gamma[i])
##    end
#end
#
#
#
#
#
#
#
#
#
#weights20 = T[+0.016528527770800,
#          +0.000433509592831,
#          +0.000433509592831,
#          +0.000433509592831,
#          +0.005830026358200,
#          +0.005830026358200,
#          +0.005830026358200,
#          +0.011438468178200,
#          +0.011438468178200,
#          +0.011438468178200,
#          +0.015224491336950,
#          +0.015224491336950,
#          +0.015224491336950,
#          +0.015312445862700,
#          +0.015312445862700,
#          +0.015312445862700,
#          +0.012184028838400,
#          +0.012184028838400,
#          +0.012184028838400,
#          +0.007998716016000,
#          +0.007998716016000,
#          +0.007998716016000,
#          +0.003849150907800,
#          +0.003849150907800,
#          +0.003849150907800,
#          -0.000316030248744,
#          -0.000316030248744,
#          -0.000316030248744,
#          +0.000875567150595,
#          +0.000875567150595,
#          +0.000875567150595,
#          +0.008232919594800,
#          +0.008232919594800,
#          +0.008232919594800,
#          +0.008232919594800,
#          +0.008232919594800,
#          +0.008232919594800,
#          +0.002419516770245,
#          +0.002419516770245,
#          +0.002419516770245,
#          +0.002419516770245,
#          +0.002419516770245,
#          +0.002419516770245,
#          +0.012902453267350,
#          +0.012902453267350,
#          +0.012902453267350,
#          +0.012902453267350,
#          +0.012902453267350,
#          +0.012902453267350,
#          +0.004235545527220,
#          +0.004235545527220,
#          +0.004235545527220,
#          +0.004235545527220,
#          +0.004235545527220,
#          +0.004235545527220,
#          +0.009177457053150,
#          +0.009177457053150,
#          +0.009177457053150,
#          +0.009177457053150,
#          +0.009177457053150,
#          +0.009177457053150,
#          +0.000352202338954,
#          +0.000352202338954,
#          +0.000352202338954,
#          +0.000352202338954,
#          +0.000352202338954,
#          +0.000352202338954,
#          +0.005056342463750,
#          +0.005056342463750,
#          +0.005056342463750,
#          +0.005056342463750,
#          +0.005056342463750,
#          +0.005056342463750,
#          +0.001786954692975,
#          +0.001786954692975,
#          +0.001786954692975,
#          +0.001786954692975,
#          +0.001786954692975,
#          +0.001786954692975]
#
#r20 = T[+0.3333333333333,
#          +0.5009504643520,
#          -0.0019009287044,
#          +0.5009504643520,
#          +0.4882129579350,
#          +0.0235740841305,
#          +0.4882129579350,
#          +0.4551366869500,
#          +0.0897266360994,
#          +0.4551366869500,
#          +0.4019962593180,
#          +0.1960074813630,
#          +0.4019962593180,
#          +0.2558929097590,
#          +0.4882141804810,
#          +0.2558929097590,
#          +0.1764882559950,
#          +0.6470234880100,
#          +0.1764882559950,
#          +0.1041708553370,
#          +0.7916582893260,
#          +0.1041708553370,
#          +0.0530689638409,
#          +0.8938620723180,
#          +0.0530689638409,
#          +0.0416187151960,
#          +0.9167625696080,
#          +0.0416187151960,
#          +0.0115819214068,
#          +0.9768361571860,
#          +0.0115819214068,
#          +0.6064026461060,
#          +0.0487415836648,
#          +0.3448557702290,
#          +0.0487415836648,
#          +0.3448557702290,
#          +0.6064026461060,
#          +0.6158426144570,
#          +0.0063141159486,
#          +0.3778432695950,
#          +0.0063141159486,
#          +0.3778432695950,
#          +0.6158426144570,
#          +0.5590480003900,
#          +0.1343165205470,
#          +0.3066354790620,
#          +0.1343165205470,
#          +0.3066354790620,
#          +0.5590480003900,
#          +0.7366067432630,
#          +0.0139738939624,
#          +0.2494193627750,
#          +0.0139738939624,
#          +0.2494193627750,
#          +0.7366067432630,
#          +0.7116751422870,
#          +0.0755491329098,
#          +0.2127757248030,
#          +0.0755491329098,
#          +0.2127757248030,
#          +0.7116751422870,
#          +0.8614027171550,
#          -0.0083681532082,
#          +0.1469654360530,
#          -0.0083681532082,
#          +0.1469654360530,
#          +0.8614027171550,
#          +0.8355869579120,
#          +0.0266860632587,
#          +0.1377269788290,
#          +0.0266860632587,
#          +0.1377269788290,
#          +0.8355869579120,
#          +0.9297561715570,
#          +0.0105477192941,
#          +0.0596961091490,
#          +0.0105477192941,
#          +0.0596961091490,
#          +0.9297561715570]
#
#    s20 = T[ +0.3333333333333,
#           +0.5009504643520,
#           +0.5009504643520,
#           -0.0019009287044,
#           +0.4882129579350,
#           +0.4882129579350,
#           +0.0235740841305,
#           +0.4551366869500,
#           +0.4551366869500,
#           +0.0897266360994,
#           +0.4019962593180,
#           +0.4019962593180,
#           +0.1960074813630,
#           +0.2558929097590,
#           +0.2558929097590,
#           +0.4882141804810,
#           +0.1764882559950,
#           +0.1764882559950,
#           +0.6470234880100,
#           +0.1041708553370,
#           +0.1041708553370,
#           +0.7916582893260,
#           +0.0530689638409,
#           +0.0530689638409,
#           +0.8938620723180,
#           +0.0416187151960,
#           +0.0416187151960,
#           +0.9167625696080,
#           +0.0115819214068,
#           +0.0115819214068,
#           +0.9768361571860,
#           +0.3448557702290,
#           +0.6064026461060,
#           +0.0487415836648,
#           +0.3448557702290,
#           +0.6064026461060,
#           +0.0487415836648,
#           +0.3778432695950,
#           +0.6158426144570,
#           +0.0063141159486,
#           +0.3778432695950,
#           +0.6158426144570,
#           +0.0063141159486,
#           +0.3066354790620,
#           +0.5590480003900,
#           +0.1343165205470,
#           +0.3066354790620,
#           +0.5590480003900,
#           +0.1343165205470,
#           +0.2494193627750,
#           +0.7366067432630,
#           +0.0139738939624,
#           +0.2494193627750,
#           +0.7366067432630,
#           +0.0139738939624,
#           +0.2127757248030,
#           +0.7116751422870,
#           +0.0755491329098,
#           +0.2127757248030,
#           +0.7116751422870,
#           +0.0755491329098,
#           +0.1469654360530,
#           +0.8614027171550,
#           -0.0083681532082,
#           +0.1469654360530,
#           +0.8614027171550,
#           -0.0083681532082,
#           +0.1377269788290,
#           +0.8355869579120,
#           +0.0266860632587,
#           +0.1377269788290,
#           +0.8355869579120,
#           +0.0266860632587,
#           +0.0596961091490,
#           +0.9297561715570,
#           +0.0105477192941,
#           +0.0596961091490,
#           +0.9297561715570,
#           +0.0105477192941]
