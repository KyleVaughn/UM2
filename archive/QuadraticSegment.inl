// -------------------------------------------------------------------
// point_is_left
// -------------------------------------------------------------------

template <length_t K, length_t P, length_t N, length_t D, typename T>
UM2_PURE UM2_HOSTDEV constexpr
bool Polytope<K, P, N, D, T>::point_is_left(Point<D, T> const & p) const
requires (K == 1 && D == 2)
{
    if constexpr (P == 1) { // LineSegment
        LineSegment<2, T> const & L = *this;
        return are_CCW(L[0], L[1], p);
    } else if constexpr (P == 2) { // QuadraticSegment
        using namespace std; // Make math functions less verbose
        // If the point is in the bounding box of the segment,
        // we need to check if the point is left of the segment.
        // To do this we must find the point on q that is closest to P.
        // At this Q(r) we compute Q'(r) × (P - Q(r)). If this quantity is
        // positive, then P is left of the segment.
        //
        // To compute Q_nearest, we find r which minimizes ‖P - Q(r)‖.
        // This r also minimizes ‖P - Q(r)‖².
        // It can be shown that this is equivalent to finding the minimum of the
        // quartic function
        // ‖P - Q(r)‖² = f(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
        // The minimum of f(r) occurs when f′(r) = ar³ + br² + cr + d = 0, where
        // 𝘄 = P - P₁
        // a = 2(𝗮 ⋅ 𝗮)
        // b = 3(𝗮 ⋅ 𝗯)
        // c = [(𝗯  ⋅ 𝗯) - 2(𝗮 ⋅𝘄)]
        // d = -(𝗯 ⋅ 𝘄)
        // Lagrange's method is used to find the roots.
        // (https://en.wikipedia.org/wiki/Cubic_equation#Lagrange's_method)
        QuadraticSegment<2, T> const & Q = *this;
        Vec2<T> const v13 = Q[2] - Q[0];
        Vec2<T> const v23 = Q[2] - Q[1];
        Vec2<T> const va = -2 * (v13 + v23);
        Vec2<T> const vb = 3 * v13 + v23;
        // Bounding box check
        Vec2<T> const vr = -vb / (2 * va);
        Vec2<T> vmin = min(Q[0], Q[1]);
        Vec2<T> vmax = max(Q[0], Q[1]);
        if (0 <= vr[0] && vr[0] <= 1) {
          T const x_stationary = vr[0] * vr[0] * va[0] + vr[0] * vb[0] + Q[0][0];
          vmin[0] = std::min(vmin[0], x_stationary);
          vmax[0] = std::max(vmax[0], x_stationary);
        }
        if (0 <= vr[1] && vr[1] <= 1) {
          T const y_stationary = vr[1] * vr[1] * va[1] + vr[1] * vb[1] + Q[0][1];
          vmin[1] = std::min(vmin[1], y_stationary);
          vmax[1] = std::max(vmax[1], y_stationary);
        }
        T const a = 2 * norm2(va);

        // If the point is outside of the box, or |a| is small, then we can
        // treat the segment as a line segment.
        bool const in_box = vmin[0] <= p[0] && p[0] <= vmax[0] &&
                            vmin[1] <= p[1] && p[1] <= vmax[1];
        if (!in_box || a < static_cast<T>(1e-7)) {
            return 0 <= cross(Q[1] - Q[0], p - Q[0]);
        }

        T const b = 3 * dot(va, vb);

        Vec2<T> const vw = p - Q[0];

        T const d = -dot(vb, vw);
//        if (d < static_cast<T>(1e-7)) { // one root is 0
//            return 0 <= cross(Q[1] - Q[0], p - Q[0]);
//        }

        T const c = (norm2(vb) - 2 * dot(va, vw));

        // Lagrange's method
        T const s0 = -b / a;
        T const e1 = s0;
        T const e2 = c / a;
        T const e3 = -d / a;
        T const A = 2 * e1 * e1 * e1 - 9 * e1 * e2 + 27 * e3;
        T const B = e1 * e1 - 3 * e2;
        T const disc = A * A - 4 * B * B * B;
        if (0 < disc) { // One real root
            const T s1 = cbrt((A + sqrt(disc)) / 2);
            T const s2 = abs(s1) < static_cast<T>(1e-7) ? 0 : B / s1;
            T const r = (s0 + s1 + s2) / 3;
            if (0 <= r && r <= 1) {
                return 0 <= cross(Q.jacobian(r), p - Q(r));
            } else {
                return 0 <= cross(Q[1] - Q[0], p - Q[0]);
            } 
        } else {
            complex<T> const t1 =
                exp(
                    static_cast<T>(0.3333333333333333) * log(
                        static_cast<T>(0.5) * (A + sqrt(static_cast<complex<T>>(disc)))
                    )
                );
            complex<T> const t2 = t1 == static_cast<complex<T>>(0) ?
                static_cast<complex<T>>(0) : B / t1; 
            // The constructor for complex<T> is constexpr, but sqrt is not.
            // zeta1 = (-1/2, sqrt(3)/2)
            constexpr complex<T> zeta1(static_cast<T>(-0.5),
                                       static_cast<T>(0.8660254037844386));
            constexpr complex<T> zeta2(conj(zeta1));

            // Pick the point closest to p
            T r = 0;
            T dist = INF_POINT<T>;

            T const r1 = real((s0 + t1 + t2)) / 3;
            if (0 <= r1 && r1 <= 1) {
                T const d1 = distance2(p, Q(r1));
                if (d1 < dist) {
                    dist = d1;
                    r = r1;
                }
            }

            T const r2 = real((s0 + zeta2 * t1 + zeta1 * t2)) / 3;
            if (0 <= r2 && r2 <= 1) {
                T const d2 = distance2(p, Q(r2));
                if (d2 < dist) {
                    dist = d2;
                    r = r2;
                }
            }

            T const r3 = real((s0 + zeta1 * t1 + zeta2 * t2)) / 3;
            if (0 <= r3 && r3 <= 1) {
                T const d3 = distance2(p, Q(r3));
                if (d3 < dist) {
                    r = r3;
                }
            }
            UM2_ASSERT(0 <= r && r <= 1);
            return 0 <= cross(Q.jacobian(r), p - Q(r));
        }
    } else {
        static_assert(!K, "Unsupported Polytope");
    }
}

// -------------------------------------------------------------------
// QuadraticSegment2 only
// -------------------------------------------------------------------

template<typename T>
UM2_PURE UM2_HOSTDEV constexpr
T enclosed_area_quadratic_segment(Point2<T> const & p0,
                                  Point2<T> const & p1,
                                  Point2<T> const & p2)
{
    // The area bounded by Q and the line from P₁ to P₂ is 4/3 the area of the
    // triangle formed by the vertices. Assumes the area is convex.
    // 1-based indexing P₁, P₂, P₃
    // Easily derived by transforming Q such that P₁ = (0, 0) and P₂ = (x₂, 0).
    // However, vertices are CCW order, so sign of the area is flipped.
    return static_cast<T>(0.6666666666666666) * cross(p2 - p0, p1 - p0);
}                                                                              

template<typename T>
UM2_PURE UM2_HOSTDEV constexpr
Point2<T> enclosed_area_centroid_quadratic_segment(Point2<T> const & p0,
                                                   Point2<T> const & p1,
                                                   Point2<T> const & p2)
{
    // For a quadratic segment, with P₁ = (0, 0), P₂ = (x₂, 0), and P₃ = (x₃, y₃),
    // where 0 < x₂, if the area bounded by q and the x-axis is convex, it can be
    // shown that the centroid of the area bounded by the segment and x-axis
    // is given by
    // C = (3x₂ + 4x₃, 4y₃) / 10
    //
    // To find the centroid of the area bounded by the segment for a general
    // quadratic segment, we transform the segment so that P₁ = (0, 0),
    // then use a change of basis (rotation) from the standard basis to the
    // following basis, to achieve y₂ = 0.
    //
    // Let v = (v₁, v₂) = (P₂ - P₁) / ‖P₂ - P₁‖
    // u₁ = ( v₁,  v₂) = v
    // u₂ = (-v₂,  v₁)
    //
    // Note: u₁ and u₂ are orthonormal.
    //
    // The transformation from the new basis to the standard basis is given by
    // U = [u₁ u₂] = | v₁ -v₂ |
    //               | v₂  v₁ |
    //
    // Since u₁ and u₂ are orthonormal, U is unitary.
    //
    // The transformation from the standard basis to the new basis is given by
    // U⁻¹ = Uᵗ = |  v₁  v₂ |
    //            | -v₂  v₁ |
    // since U is unitary.
    //
    // Therefore, the centroid of the area bounded by the segment is given by
    // C = U * Cᵤ + P₁
    // where
    // Cᵤ = (u₁ ⋅ (3(P₂ - P₁) + 4(P₃ - P₁)), 4(u₂ ⋅ (P₃ - P₁))) / 10
    Vec2<T> const v12 = p1 - p0;
    Vec2<T> const four_v13 = 4 * (p2 - p0);
    Vec2<T> const u1 = normalize(v12);
    Vec2<T> const u2(-u1[1], u1[0]);
    Mat2x2<T> const U(u1, u2);
    Vec2<T> const Cu(dot(u1, (3 * v12 + four_v13)) / 10,
                     dot(u2,            four_v13 ) / 10);
    return U * Cu + p0;
}
