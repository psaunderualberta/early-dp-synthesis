module Synthesis

using MLJ
using Distributions
using SymbolicRegression

include("Losses.jl")
using .Losses

begin
    # Dataset with two named features:"
    X = (zero=fill(0.0, 10000),)

    # and one target:
    y = @. 2 * cos(X.zero * 23.5)

    # with some noise:
    y = y .+ randn(10000) .* 1e-3

    # Define uniform
    unif(a::T, b::T) where {T} = a < b ? rand(Uniform(a, b)) : T(NaN)
    normal(a::T, b::T) where {T} = b > 0 ? rand(Normal(a, b)) : T(NaN)

    model = SRRegressor(
        save_to_file=false,
        niterations=50,
        binary_operators=[+, -, unif, normal],
        loss_function=unbalanced_dist,
        maxdepth=3,
        complexity_of_variables=5
    )

    begin
        mach = machine(model, X, y)

        fit!(mach)
    end

    # ╔═╡ 8669029b-31c9-4317-8c6c-c84120e5c9d2
    println(report(mach))
end

end