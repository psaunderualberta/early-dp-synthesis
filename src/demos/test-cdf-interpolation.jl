using Gadfly
using StatsBase
using Interpolations
using FiniteDifferences
using Distributions: std, mean
using JET
import FromFile: @from

@from "../util/distributions.jl" import uniform, normal, laplace
@from "../util/plotting.jl" import save_plot


const interpolations = Dict(
    # "linear" => Gridded(Linear()),
    # "cubic" => Gridded(Cubic(Line(OnGrid()))),
    # "quadratic" => Gridded(Quadratic(Line(OnGrid()))),
    # "LinearMonotonic" => LinearMonotonicInterpolation(),
    "SteffanMonotonic" => SteffenMonotonicInterpolation(),
    # "FiniteDifferenceMonotonic" => FiniteDifferenceMonotonicInterpolation(),
    # "FritschButlandMonotonic" => FritschButlandMonotonicInterpolation(),
)

function steffen_monotonic_interpolation(data::Vector{Float64}, eval_x::Vector{Float64}; sensitivity::Float64=1.0,
        itp::Union{Nothing, Vector{Float64}}=nothing, itp_gradient::Union{Nothing, Vector{Float64}}=nothing)
    """
        Steffen Monotonic Interpolation

        Reference: Steffen, M. (1990). A Simple Method for Monotonic Interpolation in One Dimension.
    """
    data = sort(data)
    n = length(data)
    cdf = [i / n for i in 1:n]
    s = std(data)

    # Add padding to the data
    data = append!([data[1] - sensitivity], data, [data[end] + sensitivity])
    cdf = append!([0.0], cdf, [1.0])
    n = n + 2

    a = Array{Float64}(undef, n)
    b = Array{Float64}(undef, n)
    c = Array{Float64}(undef, n)
    d = Array{Float64}(undef, n)
    h = Array{Float64}(undef, n)
    s = Array{Float64}(undef, n)
    p = Array{Float64}(undef, n)
    y_prime = Array{Float64}(undef, n)

    # h_i = x_{i+1} - x_i
    # s_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)
    # p_i = (s_{i-1} * h_i + s_i * h_{i-1}) / (h_{i-1} + h_i)

    # h, s range from i = 1 to n - 1
    h[1:end-1] = data[2:end] - data[1:end-1]
    s[1:end-1] = (cdf[2:end] - cdf[1:end-1]) ./ h[1:end-1]
    # Note that 'p' ranges from i = 2 to n - 1
    p[2:end-1] = (s[1:end-2] .* h[2:end-1] + s[2:end-1] .* h[1:end-2]) ./ (h[1:end-2] + h[2:end-1])

    for i in 2:n-1
        y_prime[i] = p[i]

        if s[i] * s[i-1] <= 0
            y_prime[i] = 0.0
        elseif abs(p[i]) > 2 * abs(s[i-1]) || abs(p[i]) > 2 * abs(s[i])
            y_prime[i] = 2 * sign(s[i]) * min(abs(s[i]), abs(s[i-1]))
        end
    end

    # Endpoints
    p[1] = s[1] * (1 + h[1] / (h[1] + h[2])) - s[2] * h[1] / (h[1] + h[2])
    p[end] = s[end-1] * (1 + h[end-1] / (h[end-1] + h[end-2])) - s[end-2] * h[end-1] / (h[end-1] + h[end-2])

    y_prime[1] = p[1]
    if p[1] * s[1] <= 0
        y_prime[1] = 0.0
    elseif abs(p[1]) > 2 * abs(s[1])
        y_prime[1] = 2 * s[1]
    end

    y_prime[end] = p[end]
    if p[end] * s[end-1] <= 0
        y_prime[end] = 0.0
    elseif abs(p[end]) > 2 * abs(s[end-1])
        y_prime[end] = 2 * s[end-1]
    end

    @. a[1:end-1] = (y_prime[1:end-1] + y_prime[2:end] - 2 * s[1:end-1]) / h[1:end-1]^2
    @. b[1:end-1] = (3 * s[1:end-1] - 2 * y_prime[1:end-1] - y_prime[2:end]) / h[1:end-1]
    @. c[1:end-1] = y_prime[1:end-1]
    @. d[1:end-1] = cdf[1:end-1]
    # for i in eachindex(data)[1:end-1]
    #     a[i] = (y_prime[i] + y_prime[i+1] - 2 * s[i]) / h[i]^2
    #     b[i] = (3 * s[i] - 2 * y_prime[i] - y_prime[i+1]) / h[i]
    #     c[i] = y_prime[i]
    #     d[i] = cdf[i]
    # end

    # Create interpolation function
    itp = isnothing(itp) ? fill(0.0, length(eval_x)) : itp
    itp_gradient = isnothing(itp_gradient) ? fill(0.0, length(eval_x)) : itp_gradient

    @assert all(data[1] .<= eval_x .<= data[end])

    for i in eachindex(eval_x)
        idx = searchsortedlast(data, eval_x[i])
        idx = min(idx, n - 1)
        idx = max(idx, 1)

        dx = eval_x[i] - data[idx]
        ai, bi, ci, di = a[idx], b[idx], c[idx], d[idx]
        itp[i] += ai * dx^3 + bi * dx^2 + ci * dx + di
        itp_gradient[i] += 3 * ai * dx^2 + 2 * bi * dx + ci
    end

    itp, itp_gradient
end

function steffen_monotonic_interpolation_bootstrap(data::AbstractVector{<:Real}, eval_x::AbstractVector{<:Real}; bootstrap_frac::Float64=0.01, num_samples::Int=100)
    @assert 0 < bootstrap_frac <= 1
    @assert num_samples > 0

    n = length(data)
    data = sort(data)
    itp = fill(0.0, size(eval_x)...)
    itp_gradient = fill(0.0, size(eval_x))
    for _ in 1:num_samples
        idx = sample(2:n-1, floor(Int, n * bootstrap_frac), replace=true)
        bootstrap_data = append!([data[1]], data[idx], [data[end]])
        itp, itp_gradient = steffen_monotonic_interpolation(bootstrap_data, eval_x; itp=itp, itp_gradient=itp_gradient)
    end

    itp / num_samples, itp_gradient / num_samples
end

function interpolate_data(data::AbstractVector{<:Real}, eval_x::AbstractVector{<:Real})
    """
        Interpolate the data using the specified interpolation method.
        We estimate the cdf of the data, interpolate that, then take the derivative.
    """

    return steffen_monotonic_interpolation_bootstrap(data, eval_x)
end

function plot_bandwidths()
    """
        Plot the kernel density estimate of the data x_i with bandwidth h.
    """

    n = 10000

    distributions = Dict(
        "Uniform(0, 1)" => :(uniform(0, 1)),
        "Uniform(-5, 5)" => :(uniform(-5, 5)),
        "Normal(0, 0.00167)" => :(normal(0.0, 0.00167)),
        "Laplace(Unif(-5, 5), 1)" => :(laplace(uniform(-5, 5), 1)),
    )

    cdf_plots = []
    pdf_plots = []
    for (name, dist) in distributions
        x = [eval(dist) for _ in 1:n]
        x_min = minimum(x)
        x_max = maximum(x)
        eval_x = collect(x_min-1:std(x)/100:x_max+1)
        interp, interp_gradient = interpolate_data(x, eval_x)

        # pcdf = plot(x=eval_x, y=interp, Geom.line, Guide.xlabel("X"), Guide.ylabel("Density"), Guide.title(name))
        # ppdf = plot(x=eval_x, y=interp_gradient, Geom.line, Guide.xlabel("X"), Guide.ylabel("CDF"), Guide.title(name))
        # push!(cdf_plots, pcdf)
        # push!(pdf_plots, ppdf)
    end

    # dir = joinpath("plots", "cdf-interpolation")
    # isdir(dir) || mkpath(dir)
    # for (name, plots) in [("pdf", pdf_plots), ("cdf", cdf_plots)]
    #     stack = gridstack(convert(Matrix{Plot}, reshape(plots, 1, 1)))
    #     save_plot(stack, joinpath(dir, "$(interp_name)-$(name).pdf"))
    # end
end

# plot_bandwidths()
