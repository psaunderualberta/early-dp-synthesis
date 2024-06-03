using Distributed

# Precompile in main thread
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

# @everywhere begin
#     using Pkg
#     Pkg.activate(joinpath(@__DIR__, ".."))
#     Pkg.instantiate()

using MLJ
using Distributions
using SymbolicRegression
using ArgParse

include("Losses.jl")

function main(args)
    accest = haskey(accuracy_estimators, args["accuracy"]) ? accuracy_estimators[args["accuracy"]] : error("A")
    privest = haskey(privacy_estimators, args["privacy"]) ? privacy_estimators[args["privacy"]] : error("B")
    combest = haskey(combiners, args["combiner"]) ? combiners[args["combiner"]] : error("C")

    # Dataset with two named features
    n = 1000

    # Ablee to use variables as keys
    d = Dict(
        "zero" => zeros(n),
        SENSITIVITY_COLUMN_NAME => fill(args["sensitivity"], n),
    )
    X =  NamedTuple(((Symbol(key), value) for (key, value) in d))

    y = @. zeros(Float64, n)

    # Define uniform
    unif(a::T, b::T) where {T} = a < b ? rand(Uniform(a, b)) : T(NaN)
    # normal(a, b) = b > 0 ? rand(Normal(a, b)) : NaN
    # laplace(b) = b > 0 ? rand(Laplace(0, b)) : NaN

    # Define loss function
    loss(tree, dataset, options) = privacy_loss(accest(), privest(), combest(), tree, dataset, options)
    processes = nprocs() > 1 ? procs()[2:end] : procs()

    # Define model
    model = SRRegressor(;
        save_to_file=false,
        niterations=5,
        binary_operators=[+, -, unif],
        unary_operators=[],
        loss_function=loss,
        maxdepth=3,
    )

    # Train model
    mach = machine(model, X, y)
    fit!(mach)

    # Print report
    println(report(mach))
end


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--accuracy"
            help = "The method of computing the accuracy of a distribution"
            arg_type = String
            default = "quadratic"
        "--privacy"
            help = "The method of computing the privacy loss for a distribution"
            arg_type = String
            default = "none"
        "--combiner"
            help = "The method of combining the accuracy & privacy loss into a single value"
            arg_type = String
            default = "linear"
        "--sensitivity"
            help = "The sensitivity of the dataset"
            arg_type = Float64
            default = 1.0
    end

    return parse_args(s)
end

args = parse_commandline()
main(args)
