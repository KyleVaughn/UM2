using MOCNeutronTransport
using Plots
pyplot()
T = Float64
Nsoln = 20
Nset = 1:10 
f = in

p₁ = Point_2D(T, 0)
p₂ = Point_2D(T, 2)
p₃ = Point_2D(T, 2, 2)
p₄ = Point_2D(T, 3//2, 1//4)
p₅ = Point_2D(T, 3, 1)
p₆ = Point_2D(T, 1, 1)
tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))
p = Point_2D(T, -1e-6, -1e-6)

soln = f(p, tri6; N=Nsoln)
err = [f(p, tri6; N = n) for n = Nset]
plot(Nset[1:length(Nset)-1], 
     err[1:length(Nset)-1], 
     title = "Quadrature Points vs Triangle6_2D ∈  error (Nsoln = $Nsoln)") 
xlabel!("Quadrature Points")
ylabel!("∈  error")
