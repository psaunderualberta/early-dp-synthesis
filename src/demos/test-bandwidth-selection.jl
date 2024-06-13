using Gadfly
using StatsBase
using SpecialFunctions
using FiniteDifferences
import FromFile: @from
import Random: seed!

@from "../util/distributions.jl" import uniform, normal, laplace
@from "../util/plotting.jl" import save_plot

const kernel_antiderivatives = Dict(
    "gaussian" => (x, x_i, h) -> h / 2 * (erf((x - x_i) / (sqrt(2) * h)) + 1),
    "laplace" => (x, x_i, h) -> begin
        if x <= x_i
            return h / 2 * exp((x - x_i) / h)
        end

        h - h / 2 * exp((x_i - x) / h)
    end,
    "uniform" => (x, x_i, h) -> begin
        if x - x_i <= - h
            return 0.0
        elseif abs(x - x_i) <= h
            return (x - x_i + h) / 2
        end

        return h
    end,
)

const sensitivity = 1.0
const distributions = Dict(
    "Uniform(0, 1)" => :(uniform(0.0, 1.0)),
    "Laplace(0, 0.05)" => :(laplace(0.0, 0.05)),
    "Laplace(Uniform(-5, 5), 1)" => :(laplace(uniform(-5, 5), 1.0)),
    "Normal(Laplace(10), Normal(50 - sensitivity, sensitivity))" => :(normal(
        laplace(10),
        normal(
            50.0 - sensitivity,
            sensitivity
        ))),
)

const smoothers = Dict(
    # bootstrapping

    # smoothing
)

function cdf(data::AbstractVector{<:Real})
    """
        Compute the empirical cdf of the data.
    """
    n = length(data)
    return sort(data), [i / n for i in 1:n]
end


function fit_kde(data::AbstractVector{<:Real}, frac::Float64; h::Union{Real, Nothing}=nothing, k::String="gaussian", seed::Union{Int,Nothing}=nothing)
    """
        Fit a KDE to the data.
    """

    @assert 0 < frac <= 1 "Fraction must be between 0 and 1"

    if !isnothing(seed)
        seed!(seed)
    end

    # Select a random subset of the data
    n = length(data)
    idxs = [i for i in eachindex(data)]
    indices = sample(idxs, floor(Int, n * frac), replace=false)
    
    kernel = kernel_antiderivatives[k]
    
    kde(x, h) = sum([kernel(x, data[i], h) for i in indices]) / (length(indices) * h)
    kde(x) = sum([kernel(x, data[i], h) for i in indices]) / (length(indices) * h)
    return kde
end

function fit_H(data::AbstractVector{<:Real}, frac::Float64; k::String="gaussian", seed::Union{Int,Nothing}=nothing)
    """
        Fit a KDE to the data.
    """

    x, y = cdf(data)

    kde = fit_kde(x, frac; k=k, seed=seed)
    H(h) = 1 / 2 * sum([(y[i] - kde(x[i], h))^2 for i in eachindex(x)])
    return H
end

function plot_cdf_fits() 
    nsamples = 1000
    hs = collect(0.1:0.2:2)
    eval_hs = collect(0.1:0.01:2.0)

    n = 2
    m = 2
    @assert n * m == length(distributions)

    firstorder = forward_fdm(5, 1)  # forward, since h < 0 is not defined.
    secondorder = forward_fdm(5, 2)

        
    dir = joinpath("plots", "interpolation")
    isdir(dir) || mkpath(dir)
    for kernel in keys(kernel_antiderivatives)
        cdfplots = []
        hplots = []
        fitplots = []
        pdfplots = []
    
        for (name, dist) in distributions
            x = [eval(dist) for _ in 1:nsamples]
            x, y = cdf(x)
    
            kernels = [fit_kde(x, 0.1; h=h, seed=42, k=kernel) for h in hs]
    
            # Construct data matrix
            @time data = stack([x, y, [k.(x) for k in kernels]...])
    
            # 
            p = plot(
                data, x=Col.value(1), y=Col.value(2:size(data, 2)...), color=Col.index(2:size(data, 2)...),
                Geom.line,
                Scale.color_discrete,
                Guide.colorkey(labels=append!(["true"], map(string, hs))),
                Guide.title(name),
                Guide.xlabel("x"),
                Guide.ylabel("F(x)")
            )
            push!(cdfplots, p)
    
            H = fit_H(x, 0.1; seed=42, k=kernel)
            p = plot(x=eval_hs, y=[H(h) for h in eval_hs], Geom.line, Guide.xlabel("Bandwidth"), Guide.ylabel("H"), Guide.title(name))
            push!(hplots, p)
    
            # Perform newton's method to find the optimal bandwidth
            h = 0.05
            fits = [fit_kde(x, 0.1; h=h, seed=42, k=kernel).(x)]
            @time firstorder(H, h) / secondorder(H, h)
            for _ in 1:10
                h = h - firstorder(H, h) / secondorder(H, h)
                h = max(1e-3, h)
                push!(fits, fit_kde(x, 0.1; h=h, seed=42, k=kernel).(x))
            end
    
            data = stack([x, y, fits...])
            p = plot(
                data, x=Col.value(1), y=Col.value(2:size(data, 2)...), color=Col.index(2:size(data, 2)...),
                Geom.line,
                Scale.color_discrete,
                Guide.colorkey(labels=append!(["true"], map(string, 1:length(fits)))),
                Guide.title(name),
                Guide.xlabel("x"),
                Guide.ylabel("F(x)")
            )
            push!(fitplots, p)

            # Compute the pdf
            points = collect(-2:0.01:2)
            kde = fit_kde(x, 0.1; h=h, seed=42, k=kernel)
            pdf = [central_fdm(5, 1; factor=1e-6)(kde, x) for x in points]
            data = stack([points, pdf])  # No clue why I need to use 'stack' here rather than pass in normally
            p = plot(data, x=Col.value(1), y=Col.value(2), Geom.line, Guide.xlabel("x"), Guide.ylabel("f(x)"), Guide.title(name))
            push!(pdfplots, p)
        end

        for (fname, plots) in [("$(kernel)-cdf", cdfplots), ("$(kernel)-h", hplots), ("$(kernel)-fit", fitplots), ("$(kernel)-pdf", pdfplots)]
            mat = convert(Matrix{Plot}, reshape(plots, n, m))
            composite = gridstack(mat)
    
            save_plot(composite, joinpath(dir, "$fname.pdf"))
        end
    end
end

function plot_bandwidths()
    """
        Plot the kernel density estimate of the data x_i with bandwidth h.
    """

    n = 10_000
    hs = collect(0.05:0.01:20.0)

    plots = []
    for (name, dist) in distributions
        x = [eval(dist) for _ in 1:n]

        p = plot(x=hs, y=[H(KDE(x, h), test_hist, h) for h in hs], Geom.line, Guide.xlabel("Bandwidth"), Guide.ylabel("H"), Guide.title(name))
        push!(plots, p)
    end


    mat = convert(Matrix{Plot}, reshape(plots, 2, 2))
    return gridstack(mat)
end

function test_bandwidth_estimation()
    n = 10000

    for (name, dist) in distributions
        x = [eval(dist) for _ in 1:n]

        fn = kde(x; boundary=(minimum(x), maximum(x)))
        ik = InterpKDE(fn)  
    end
end

plot_cdf_fits()
