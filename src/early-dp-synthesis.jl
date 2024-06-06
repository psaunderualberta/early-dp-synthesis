using Distributed

# Precompile in main thread
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.resolve()
Pkg.instantiate()

@everywhere begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.resolve()
    Pkg.instantiate()
end

@everywhere begin
    using MLJ
    using Distributions
    using SymbolicRegression
    using DynamicExpressions: Node
    using ArgParse
    using Dates
    import FromFile: @from
end

@everywhere begin
    @from "Common.jl" import accuracy_estimators, privacy_estimators, combiners
    @from "Constants.jl" import SENSITIVITY_COLUMN_NAME
    @from "./util/losses.jl" import privacy_loss
    @from "./util/dataset.jl" import create_dataset
    @from "./util/distributions.jl" import normal, uniform, laplace
    @from "./util/plotting.jl" import kde_density_plot, save_plot, PLOTTING_FUNCTIONS
    @from "./util/simplification.jl" import insert_variables

    function main(args)
        println("Running with args: ")
        display(args)

        # Define the methods of computing accuracy, privacy, and combining them
        accest = haskey(accuracy_estimators, args["accuracy"]) ? accuracy_estimators[args["accuracy"]] : error("A")
        privest = haskey(privacy_estimators, args["privacy"]) ? privacy_estimators[args["privacy"]] : error("B")
        combest = haskey(combiners, args["combiner"]) ? combiners[args["combiner"]] : error("C")

        # Create a standard dataset
        X, y, vars = create_dataset(args["n_samples"])

        # Define loss function
        loss(tree, dataset, options) = privacy_loss(accest(), privest(), combest(), tree, dataset, options)
        processes = nprocs() > 1 ? procs()[2:end] : procs()

        # Define output file, creating dirs if necessary
        if args["save"]
            args["outpath"] = joinpath(args["outpath"], args["data-subdir"])
            args["plotpath"] = joinpath(args["plotpath"], args["data-subdir"])
            args["outpath"] = isdir(args["outpath"]) ? args["outpath"] : mkpath(args["outpath"])
            args["plotpath"] = isdir(args["plotpath"]) ? args["plotpath"] : mkpath(args["plotpath"])
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
            binary_operators=[+, -, normal, uniform],
            unary_operators=[laplace],
            loss_function=loss,
            maxdepth=10,
            progress=true,
        )

        # Train model
        mach = machine(model, X, y)
        fit!(mach)  # Not sure why this says 'possible method call error'

        # Print report
        rep = report(mach)
        println(rep)
        nowtime = Dates.format(now(), "yyyy-mm-dd-HH-MM")
        if args["save"]
            # Generate plots
            if isempty(args["plots"])
                args["plots"] = PLOTTING_FUNCTIONS
            end

            num_equations = length(rep.equations)
            println(num_equations)
            for i in 1:num_equations
                equation = rep.equation_strings[i]  # TODO: Where are the operators stored?
                complexity = rep.complexities[i]

                # Sample from the equation
                quoted_equation = Meta.parse(insert_variables(equation, vars))
                samples = [eval(quoted_equation) for _ in 1:args["n_samples"]]

                for (plotname, plotfunc) in args["plots"]
                    println("Plotting '$plotname' for '$equation'")
                    filename = joinpath(args["plotpath"], "$(complexity)-$(plotname).pdf")
                    p = plotfunc(equation, samples)
                    save_plot(p, filename)
                end
            end
        end

        return rep
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
        "--n_samples", "-n"
            help = "The number of samples to generate when creating the dataset"
            arg_type = Int
            default = 10000
            range_tester = x -> x > 0
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
        "--data-subdir"
            help = "The subdirectory to write plots & logs within, in their respective directories. '--save' must be set to true for this to take effect."
            arg_type = String
            default = "$(nowtime)"
        "--plots"
            help = "The specific plots to generate. If not set, all plots will be generated. '--save' must be set to true for this to take effect."
            arg_type = String
            action = :store_arg
            nargs = '*'
            range_tester = x -> all(in(f, PLOTTING_FUNCTIONS) for f in x)
        "--outfile"
            help = "The file to write the output of synthesis to. '--save' must be set to true for this to take effect."
            arg_type = String
            default = "$(nowtime)-output.log"
    end

    return parse_args(s)
end

args = parse_commandline()
main(args)
