using MOCNeutronTransport
using Plots
pyplot()
T = Float64
Nsoln = 79
Nset = [12, 27, 48, 79] 
f = area

p₁ = Point_2D(T, 0)
p₂ = Point_2D(T, 2)
p₃ = Point_2D(T, 2, 2)
p₄ = Point_2D(T, 3//2, 1//4)
p₅ = Point_2D(T, 3, 1)
p₆ = Point_2D(T, 1, 1)
tri6 = Triangle6_2D((p₁, p₂, p₃, p₄, p₅, p₆))

soln = f(tri6; N=Nsoln)
err = [abs(soln - f(tri6; N = n)) for n = Nset]
plot(Nset[1:length(Nset)-1], 
     err[1:length(Nset)-1], 
     title = "Quadrature Points vs Triangle6_2D area error (Nsoln = $Nsoln)") 
xlabel!("Quadrature Points")
ylabel!("area error")
