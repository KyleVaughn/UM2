isplanar(tri::Triangle3D) = true
function isplanar(quad::Quadrilateral3D)
    # If the surface normals of two triangles, composed of vertices (1,2,3) and (1,3,4), 
    # face the same direction, the quad is planar
    𝗻₁ = (quad[2] - quad[1]) × (quad[3] - quad[1])
    𝗻₂ = (quad[4] - quad[1]) × (quad[3] - quad[1])
    return norm(𝗻₁ ⋅ 𝗻₂) ≈ norm(𝗻₁) * norm(𝗻₂)
end
