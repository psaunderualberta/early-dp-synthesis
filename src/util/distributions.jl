using Distributions

# Define uniform
uniform(a::T, b::T) where {T} = a < b ? rand(Uniform(a, b)) : T(NaN)
normal(a::T, b::T) where {T} = b > 0 ? rand(Normal(a, b)) : NaN
laplace(b) = b > 0 ? rand(Laplace(0, b)) : NaN