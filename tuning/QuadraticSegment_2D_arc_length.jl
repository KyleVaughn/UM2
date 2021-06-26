using MOCNeutronTransport
using Plots
pyplot()
T = Float64
Nsoln = 20
Nset = [ 1, 2, 3, 4, 5, 10, 15, 20]
f = arc_length

p₁ = Point_2D(T, 0)
p₂ = Point_2D(T, 2)
p₃ = Point_2D(T, 1.5, 1)
q = QuadraticSegment_2D(p₁, p₂, p₃)

soln = f(q; N=Nsoln)
err = [abs(soln - f(q; N = n)) for n = Nset]
plot(Nset[1:length(Nset)-1], 
     err[1:length(Nset)-1], 
     title = "Quadrature Points vs QuadraticSegment_2D arc_length error (Nsoln = $Nsoln)", 
     yaxis=:log) 
#     xaxis=:log)
xlabel!("Quadrature Points")
ylabel!("arc_length error")
