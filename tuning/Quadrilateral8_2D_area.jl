using MOCNeutronTransport
using Plots
pyplot()
T = Float64
Nsoln = 20
Nset = [ 1, 2, 3, 4, 5, 10, 15, 20]
f = area
p₁ = Point_2D(T, 0)
p₂ = Point_2D(T, 2)
p₃ = Point_2D(T, 2, 3)
p₄ = Point_2D(T, 0, 3)
p₅ = Point_2D(T, 3//2, 1//2)
p₆ = Point_2D(T, 5//2, 3//2)
p₇ = Point_2D(T, 1//2, 5//2)
p₈ = Point_2D(T, 0,    3//2)
quad8 = Quadrilateral8_2D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))

soln = f(quad8; N=Nsoln)
err = [abs(soln - f(quad8; N = n)) for n = Nset]
plot(Nset[1:length(Nset)-1] .* Nset[1:length(Nset)-1], 
     err[1:length(Nset)-1], 
     title = "Quadrature Points vs Quadrilateral8_2D area error (Nsoln = $Nsoln)",
     xaxis=:log,
     yaxis=:log
    ) 
xlabel!("Quadrature Points")
ylabel!("area error")
