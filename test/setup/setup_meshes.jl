function setup_TriangleMesh(::Type{T}, ::Type{U}) where {T, U}
            points = [Point2D{F}(0, 0), 
                      Point2D{F}(1, 0), 
                      Point2D{F}(0.5, 1), 
                      Point2D{F}(1.5, 1)] 
            faces = [SVector{3, U}(1, 2, 3), SVector{3, U}(2, 4, 3)] 
            face_sets = Dict{String, BitSet}()
            face_sets["A"] = BitSet([1])
            face_sets["B"] = BitSet([2])
            return TriangleMesh{T, U}(name, points, faces, face_sets)
end

function setup_QuadrilateralMesh(::Type{T}, ::Type{U}) where {T, U}
            points = [Point2D{F}(0, 0),
                      Point2D{F}(1, 0),
                      Point2D{F}(1, 1),
                      Point2D{F}(0, 1),
                      Point2D{F}(2, 0),
                      Point2D{F}(2, 1)]
            faces = [SVector{4,U}(1, 2, 3, 4), SVector{4,U}(2, 5, 6, 3)]
            face_sets = Dict{String, BitSet}()
            face_sets["A"] = Set([1])
            face_sets["B"] = Set([2])
            return QuadrilateralMesh{T, U}(name, points, faces, face_sets)
end
