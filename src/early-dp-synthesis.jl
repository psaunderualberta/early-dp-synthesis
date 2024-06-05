using Distributed

# Precompile in main thread
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

@everywhere begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
end

@everywhere begin
    using MLJ
    using Distributions
    using SymbolicRegression
    using ArgParse
    using Dates
    import FromFile: @from

    @from "Losses.jl" import privacy_loss
    @from "Common.jl" import accuracy_estimators, privacy_estimators, combiners
    @from "Constants.jl" import SENSITIVITY_COLUMN_NAME

    function main(args)
        println("Running with args: ")
        display(args)
        accest = haskey(accuracy_estimators, args["accuracy"]) ? accuracy_estimators[args["accuracy"]] : error("A")
        privest = haskey(privacy_estimators, args["privacy"]) ? privacy_estimators[args["privacy"]] : error("B")
        combest = haskey(combiners, args["combiner"]) ? combiners[args["combiner"]] : error("C")

        # Dataset with two named features
        n = 10000

        # Ablee to use variables as keys
        d = Dict(
            SENSITIVITY_COLUMN_NAME => fill(args["sensitivity"], n),
        )

        X =  NamedTuple(((Symbol(key), value) for (key, value) in d))
        y = @. zeros(Float64, n)

        # Define uniform
        unif(a::T, b::T) where {T} = a < b ? rand(Uniform(a, b)) : T(NaN)
        normal(a, b) = b > 0 ? rand(Normal(a, b)) : NaN
        laplace(b) = b > 0 ? rand(Laplace(0, b)) : NaN

        # Define loss function
        loss(tree, dataset, options) = privacy_loss(accest(), privest(), combest(), tree, dataset, options)
        processes = nprocs() > 1 ? procs()[2:end] : procs()

        # Define output file, creating dirs if necessary
        if args["save"]
            args["outpath"] = isdir(args["outpath"]) ? args["outpath"] : mkpath(args["outpath"])
            outfile = joinpath(args["outpath"], args["outfile"])
            touch(outfile)
            @assert isfile(outfile) "Could not create file at $(outfile)"
        else
            outfile = nothing
        end

        # Define model
        model = SRRegressor(;
            save_to_file=args["save"],
            output_file=outfile,
            parallelism=:multiprocessing,
            procs=processes,
            timeout_in_seconds=args["timeout"], # 10 minutes
            # niterations=5,
            binary_operators=[+, -, normal, unif],
            unary_operators=[laplace],
            loss_function=loss,
            maxdepth=10,
            progress=true,
        )

        # Train model
        mach = machine(model, X, y)
        fit!(mach)  # Not sure why this says 'possible method call error'

        # Print report
        println(report(mach))
    end
end


function parse_commandline()
    s = ArgParseSettings()
    nowtime = Dates.format(now(), "yyyy-mm-dd-HH-MM")
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
        "--timeout"
            help = "The maximum time to run the symbolic regression model, in seconds"
            arg_type = Int
            default = 60 # 1 minute
        "--save"
            help = "Whether to save the output of synthesis to a file"
            action = :store_true
        "--outpath"
            help = "The directory to write the output of synthesis to. '--save' must be set to true for this to take effect."
            arg_type = String
            default = "logs"
        "--plotpath"
            help = "The directory to write the plots of synthesis to. '--save' must be set to true for this to take effect."
            arg_type = String
            default = "plots"
        "--outfile"
            help = "The file to write the output of synthesis to. '--save' must be set to true for this to take effect."
            arg_type = String
            default = "$(nowtime)-output.log"
    end

    return parse_args(s)
end

args = parse_commandline()
main(args)
