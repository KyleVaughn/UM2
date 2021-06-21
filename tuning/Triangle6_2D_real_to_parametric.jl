using MOCNeutronTransport
using Plots
pyplot()
T = Float64
Nsoln = 20
Nset = 1:20
f = real_to_parametric

p₁ = Point_2D(T, 0)
p₂ = Point_2D(T, 2)
p₃ = Point_2D(T, 2, 2)
p₄ = Point_2D(T, 3//2, 1//4)
p₅ = Point_2D(T, 3, 1)
p₆ = Point_2D(T, 1, 1)
tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))
p = Point_2D(T, 100, 100)

soln = f(p, tri6; N=Nsoln)
err = [norm(tri6(soln) - tri6(f(p, tri6; N = n))) for n = Nset]
plot(Nset[1:length(Nset)-1], 
     err[1:length(Nset)-1], 
     title = "Newton-Raphson iterations vs Triangle6_2D norm of point error (Nsoln = $Nsoln)",
     xaxis=:log)
#     yaxis=:log)
xlabel!("Newton-Raphson iterations")
ylabel!("norm of point error")
