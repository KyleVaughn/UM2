using MOCNeutronTransport
using Plots
pyplot()
T = Float64
Nmax = 100
Nset = 1:Nmax


p₁ = Point_3D(T, 0)
p₂ = Point_3D(T, 2)
p₃ = Point_3D(T, 2, 3)
p₄ = Point_3D(T, 0, 3)
p₅ = Point_3D(T, 3//2, 1//2)
p₆ = Point_3D(T, 5//2, 3//2)
p₇ = Point_3D(T, 3//2, 5//2)
p₈ = Point_3D(T, 0,    3//2)
p₈ = Point_3D(T, 0, 3//2, 2)
quad8 = Quadrilateral8_3D((p₁, p₂, p₃, p₄, p₅, p₆, p₇, p₈))
l = LineSegment_3D(Point_3D(T, 1, 0, 1//2),
                   Point_3D(T, 1, 3, 1//2))

p₁ = intersect(l, quad8; N=Nmax)[3]
p₂ = intersect(l, quad8; N=Nmax)[4]

p1_error = [norm(p₁ - intersect(l, quad8; N = n)[3]) for n = Nset]
p2_error = [norm(p₂ - intersect(l, quad8; N = n)[4]) for n = Nset]
p_error = p1_error + p2_error
plot(Nset[1:Nmax-1], 
     p_error[1:Nmax-1], 
     title = "Edge Subdivisions vs Quad8_3D sum of norm of point errors (Nsoln = $Nmax)", 
     yaxis=:log) 
#     xaxis=:log)
xlabel!("Edge Subdivisions")
ylabel!("sum of 2-norm of point error vectors")
