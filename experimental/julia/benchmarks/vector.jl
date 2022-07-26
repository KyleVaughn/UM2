include("../typedefs/typedefs.jl")

function test_addition(::Type{T}) where {T <: AbstractFloat}
    name = "addition"
    trait1 = "static"
    trait2 = "dynamic"
    println(format_string(string(T)))

    static_v1 = Vec2{T}(1, 2)
    static_v2 = Vec2{T}(3, 4)
    dynamic_v1 = T[1, 2]
    dynamic_v2 = T[3, 4]

    t1 = @belapsed $static_v1 + $static_v2
    @test static_v1 + static_v2 == Vec2{T}(4, 6)

    t2 = @belapsed $dynamic_v1 + $dynamic_v2
    @test dynamic_v1 + dynamic_v2 == T[4, 6]

    display_results(name, trait1, trait2, t1, t2)
end

function test_cross(::Type{T}) where {T <: AbstractFloat}
    name = "cross"
    trait1 = "static"
    trait2 = "dynamic"
    println(format_string(string(T)))

    static_v1 = Vec2{T}(1, 2)
    static_v2 = Vec2{T}(3, 4)
    dynamic_v1 = T[1, 2]
    dynamic_v2 = T[3, 4]

    t1 = @belapsed cross($static_v1, $static_v2)
    @test abs(cross(static_v1, static_v2) + 2) < 1e-6

    t2 = @belapsed cross($dynamic_v1, $dynamic_v2)
    @test abs(cross(static_v1, static_v2) + 2) < 1e-6

    display_results(name, trait1, trait2, t1, t2)
end

for (func, func_name) in [(test_addition, "Vector addition"),
                          (test_cross, "Vector cross product")]
    println(func_name)
    for T in (Float16, Float32, Float64)
        func(T)
    end
end
