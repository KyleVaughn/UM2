include("../types/types.jl")

function test_addition(::Type{T}) where {T <: AbstractFloat}
    name = "addition"
    trait1 = "static"
    trait2 = "dynamic"
    println(format_string(string(T)))

    x1 = T(rand(UInt64) % 10)
    y1 = T(rand(UInt64) % 10)
    x2 = T(rand(UInt64) % 10)
    y2 = T(rand(UInt64) % 10)
    static_v1 = Vec2{T}(x1, y1)
    static_v2 = Vec2{T}(x2, y2)
    dynamic_v1 = T[x1, y1]
    dynamic_v2 = T[x2, y2]

    t1 = @belapsed $static_v1 + $static_v2
    @test static_v1 + static_v2 == Vec2{T}(x1 + x2, y1 + y2)

    t2 = @belapsed $dynamic_v1 + $dynamic_v2
    @test dynamic_v1 + dynamic_v2 == T[x1 + x2, y1 + y2]

    display_results(name, trait1, trait2, t1, t2)
end

function test_cross(::Type{T}) where {T <: AbstractFloat}
    name = "cross"
    trait1 = "static"
    trait2 = "dynamic"
    println(format_string(string(T)))


    x1 = T(rand(UInt64) % 10)
    y1 = T(rand(UInt64) % 10)
    x2 = T(rand(UInt64) % 10)
    y2 = T(rand(UInt64) % 10)
    static_v1 = Vec2{T}(x1, y1)
    static_v2 = Vec2{T}(x2, y2)
    dynamic_v1 = T[x1, y1]
    dynamic_v2 = T[x2, y2]
    c = x1 * y2 - x2 * y1

    t1 = @belapsed cross($static_v1, $static_v2)
    @test abs(cross(static_v1, static_v2) - c) < 1e-6

    t2 = @belapsed cross($dynamic_v1, $dynamic_v2)
    @test abs(cross(static_v1, static_v2) - c) < 1e-6

    display_results(name, trait1, trait2, t1, t2)
end

for (func, func_name) in [(test_addition, "Vector addition"),
                          (test_cross, "Vector cross product")]
    println(func_name)
    for T in (Float32, Float64)
        func(T)
    end
end
