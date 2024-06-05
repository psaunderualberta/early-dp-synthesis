using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate();

using DynamicExpressions
using Distributions

n = 100000

x = Vector{Float64}(undef, n);
x1 = zeros(n)
x2 = ones(n)

unif(a::T, b::T) where {T} = rand(Uniform(a, b))

ops = OperatorEnum(; binary_operators=[<, unif])

f1 = Node(; feature=1)
f2 = Node(; feature=2)

expression = unif(f1, f2)

expression([x1; x2], operators)

@time for i in eachindex(x)
    x[i] = unif(x1[i], x2[i])
end

@time @inbounds @simd for i in eachindex(x)
    x[i] = unif(x1[i], x2[i])
end

@assert length(unique(x)) > 1


# println(x).