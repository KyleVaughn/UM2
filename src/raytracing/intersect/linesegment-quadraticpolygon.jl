function Base.intersect(l::LineSegment{Point{2, T}},
                        poly::QuadraticPolygon{N, Point{2, T}}) where {N, T}
    return mapreduce(edge -> intersect(l, edge), vcat, edges(poly))
end

#
## A quadratic triangle, defined in 3D.
#
## Summary of intersection methods:
##       Triangulation
##           - Speed varies dramatically with precision
##               - The faster method when precision at or below 2 decimal places is desired
##           - Can provide false positives and false negatives due to approximation of the surface
##           - Constant time for a given number of triangles
##               - The only way to determine if two intersections exist is to test until two unique
##                 points are found. Since this happens infrequently, all points are tested, so time
##                 isn't wasted on control logic.
##       Newton-Raphson (iterative)
##           - Speed varies slightly based upon line segment length and orientation
##               - A shorter line segment will converge faster
##               - Converges based upon Jacobian matrix.
##                   - If derivatives are small, the iteration can become slow
##               - The faster method when precision beyond 2 decimal places is desired
##           - Less accurate
##               - May falsely give one intersection instead of two, especially for longer segments.
##               - This is due to the point of convergence being dependent on the initial guess
##                 point. The two starting points are placed close to the line segment start/stop
##                 to try to mitigate this.
##           - Precision to 6+ decimal places.
##       Overall
##           - Triangulation is predictable in speed, slow for high precision, and largely accurate
##           - Newton-Raphson is unpredictable in speed, fast for high precision, and
##               can be inaccurate for 2 intersections
##           - Consider the difference in timing between intersections for porting to GPU
##               - Newton-Raphson may cause thread divergence
#
#
## Triangulate, then intersect
#function intersect(l::LineSegment_3D{T}, tri6::Triangle6_3D{T};
#        N::Int64 = 25) where {T <: AbstractFloat}
#    triangles = triangulate(tri6, N)
#    npoints = 0
#    p??? = Point_3D(T, 0)
#    p??? = Point_3D(T, 0)
#    intersections = l .??? triangles
#    bools = map(x->x[1], intersections)
#    points = map(x->x[2], intersections)
#    npoints = count(bools)
#    p??? = Point_3D(T, 0)
#    p??? = Point_3D(T, 0)
#    if npoints === 0
#        return false, 0, p???, p???
#    elseif npoints === 1
#        p??? = points[argmax(bools)]
#        return true, 1, p???, p???
#    elseif npoints === 2
#        indices = findall(bools)
#        p??? = points[indices[1]]
#        p??? = points[indices[2]]
#        # Check uniqueness
#        if p??? ??? p???
#            return true, 1, p???, p???
#        else
#            return true, 2, p???, p???
#        end
#    else
#        # Account for 3 points and 4 points?
#        # If intersection is on edge shared by two triangles on entrance and/or exit 3/4 intersections
#        # can be detected
#        return true, -1, p???, p???
#    end
#end
#
## A more exact intersection algorithm that triangulation, uses Newton-Raphson.
#function intersect_iterative(l::LineSegment_3D{T}, tri6::Triangle6_3D{T};
#        N::Int64=30) where {T <: AbstractFloat}
#    p??? = Point_3D(T, 0)
#    p??? = Point_3D(T, 0)
#    npoints = 0
#    u??? = l.points[2] - l.points[1]
#    ray_start = real_to_parametric(l(0), tri6; N=10) # closest r,s to the ray start
#    ray_stop  = real_to_parametric(l(1), tri6; N=10) # closest r,s to the ray stop
#    # The parametric coordinates corresponding to the start of the line segment
#    r??? = ray_start[1]
#    s??? = ray_start[2]
#    t??? = T(0)
#    # The parametric coordinates corresponding to the start of the line segment
#    r??? = ray_stop[1]
#    s??? = ray_stop[2]
#    t??? = T(1)
#    # Iteration for point 1
#    err??? = tri6(r???, s???) - l(t???)
#    for i = 1:N
#        ???r???, ???s??? = derivatives(tri6, r???, s???)
#        J??? = hcat(???r???.x, ???s???.x, -u???.x)
#        # If matrix is singular, it's probably bad luck. Just perturb it a bit.
#        if abs(det(J???)) < 1e-5
#            r???, s???, t??? = [r???, s???, t???] + rand(3)/10
#        else
#            r???, s???, t??? = [r???, s???, t???] - inv(J???) * err???.x
#            err??? = tri6(r???, s???) - l(t???)
#            if norm(err??? - err???) < 5e-6
#                break
#            end
#            err??? = err???
#        end
#    end
#    # Iteration for point 2
#    err??? = tri6(r???, s???) - l(t???)
#    for j = 1:N
#        ???r???, ???s??? = derivatives(tri6, r???, s???)
#        J??? = hcat(???r???.x, ???s???.x, -u???.x)
#        if abs(det(J???)) < 1e-5
#            r???, s???, t??? = [r???, s???, t???] + rand(3)/10
#        else
#            r???, s???, t??? = [r???, s???, t???] - inv(J???) * err???.x
#            err??? = tri6(r???, s???) - l(t???)
#            if norm(err??? - err???) < 5e-6
#                break
#            end
#            err??? = err???
#        end
#    end
#
#    p??? = l(t???)
#    if (0 ??? r??? ??? 1) && (0 ??? s??? ??? 1) && (0 ??? t??? ??? 1) && (p??? ??? tri6(r???, s???))
#        npoints += 1
#    end
#
#    p??? = l(t???)
#    if (0 ??? r??? ??? 1) && (0 ??? s??? ??? 1) && (0 ??? t??? ??? 1) && (p??? ??? tri6(r???, s???))
#        npoints += 1
#        # If only point 2 is valid, return it as p???
#        # If points are duplicate, reduce npoints
#        if npoints === 2 && p??? ??? p???
#            npoints -= 1
#        elseif npoints === 1
#            p??? = p???
#        end
#    end
#    return npoints > 0, npoints, p???, p???
#end
