using MOCNeutronTransport
using Plots
pyplot()
T = Float64
Nmax = 50
Nset = 1:Nmax

p₁ = Point_3D(T, 0)
p₂ = Point_3D(T, 2, 0, 3)
p₃ = Point_3D(T, 2, 2)
p₄ = Point_3D(T, 1, 1//4)
p₅ = Point_3D(T, 3, 1)
p₆ = Point_3D(T, 1, 1)
tri6 = Triangle6_3D((p₁, p₂, p₃, p₄, p₅, p₆))
l = LineSegment_3D(Point_3D(T, 1, -15, -1//5),
                   Point_3D(T, 1,  15, -1//5))
p₁ = intersect_iterative(l, tri6)[3]
p₂ = intersect_iterative(l, tri6)[4]

p1_error = [norm(p₁ - intersect(l, tri6; N = n)[3]) for n = Nset]
p2_error = [norm(p₂ - intersect(l, tri6; N = n)[4]) for n = Nset]
p_error = p1_error + p2_error
plot(Nset[1:Nmax-1], 
     p_error[1:Nmax-1], 
     title = "Edge Subdivisions vs Triangle6_3D sum of norm of point errors (Nsoln = $Nsoln)", 
     yaxis=:log) 
#     xaxis=:log)
xlabel!("Edge Subdivisions")
ylabel!("sum of 2-norm of point error vectors")
