using MOCNeutronTransport
using Plots
Nmax = 1000

type = Float64
p₁ = Point( type(0) )
p₂ = Point( type(2), type(0), type(3) )
p₃ = Point( type(2), type(2) )
p₄ = Point( type(1), type(1)/type(4) )
p₅ = Point( type(3), type(1) )
p₆ = Point( type(1), type(1) )
tri6 = Triangle6((p₁, p₂, p₃, p₄, p₅, p₆))

l = LineSegment(Point( type(1), type(1)/type(2), type(-2)), 
                 Point( type(1), type(1)/type(2), type(2)))
p = intersect(l, tri6; N = Nmax)[3][1]
Nset = 1:Nmax
perror = [norm(p - intersect(l, tri6; N = n)[3][1]) for n = Nset]
plot(Nset[1:Nmax-1], 
     perror[1:Nmax-1], 
     title = "Edge Subdivisions vs Error for Nmax = $Nmax", 
     yaxis=:log, 
     xaxis=:log)
xlabel!("Edge Subdivisions")
ylabel!("2-norm of point error vector")
