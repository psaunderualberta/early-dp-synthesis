module Synthesis

using MLJ
using Distributions
using SymbolicRegression
using ArgParse

include("Common.jl")
using .Common

include("Losses.jl")
using .Losses

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--accuracy"
            help = "The method of computing the accuracy of a distribution"
            arg_type = String
            default = "mean"
        "--privacy"
            help = "The method of computing the privacy loss for a distribution"
            arg_type = String
            default = "none"
        "--combiner"
            help = "The method of combining the accuracy & privacy loss into a single value"
            arg_type = String
            default = "linear"
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    accest = haskey(accuracy_estimators, args["accuracy"]) ? accuracy_estimators[args["accuracy"]] : error("A")
    privest = haskey(privacy_estimators, args["privacy"]) ? privacy_estimators[args["privacy"]] : error("B")
    combest = haskey(combiners, args["combiner"]) ? combiners[args["combiner"]] : error("C")
    
    # Dataset with two named features:"
    X = (zero=fill(0.0, 10000),)

    # and one target:
    y = @. 2 * cos(X.zero * 23.5)

    # with some noise:
    y = y .+ randn(10000) .* 1e-3

    # Define uniform
    unif(a::T, b::T) where {T} = a < b ? rand(Uniform(a, b)) : T(NaN)
    normal(a::T, b::T) where {T} = b > 0 ? rand(Normal(a, b)) : T(NaN)

    loss(tree, dataset, options) = privacy_loss(accest, privest, combest, tree, dataset, options)
    model = SRRegressor(
        save_to_file=false,
        niterations=50,
        binary_operators=[+, -, unif, normal],
        loss_function=loss,
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

main()

end