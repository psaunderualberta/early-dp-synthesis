using Gadfly
using StatsBase
using Interpolations
using FiniteDifferences
using Distributions: std, mean
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

function steffen_monotonic_interpolation(data::AbstractVector{<:Real}; sensitivity::Real=1.0)
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

    a = fill(0.0, n)
    b = fill(0.0, n)
    c = fill(0.0, n)
    d = fill(0.0, n)
    y_prime = fill(0.0, n)

    # h_i = x_{i+1} - x_i
    # s_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)
    # p_i = (s_{i-1} * h_i + s_i * h_{i-1}) / (h_{i-1} + h_i)

    # h, s range from i = 1 to n - 1
    h = data[2:end] - data[1:end-1]
    s = (cdf[2:end] - cdf[1:end-1]) ./ h
    # Note that 'p' ranges from i = 2 to n - 1
    p = (s[1:end-1] .* h[2:end] + s[2:end] .* h[1:end-1]) ./ (h[1:end-1] + h[2:end])

    # Add padding to the data
    h = append!(h, h[end])
    s = append!(s, s[end])
    p = append!([0.0], p, [0.0])

    for i in 2:n-1  # @inbounds @simd 
        y_prime[i] = p[i]

        if s[i] * s[i-1] <= 0
            y_prime[i] = 0.0
        elseif abs(p[i]) > 2 * abs(s[i-1]) || abs(p[i]) > 2 * abs(s[i])
            y_prime[i] = 2 * sign(s[i]) * min(abs(s[i]), abs(s[i-1]))
        end
    end

    # Endpoints
    y_prime[1] = 3 / 2 * s[1] - y_prime[2] / 2
    y_prime[end] = 3 / 2 * s[end-1] - y_prime[end-2] / 2

    for i in eachindex(data)[1:end-1]  # @inbounds @simd 
        a[i] = (y_prime[i] + y_prime[i+1] - 2 * s[i]) / h[i]^2
        b[i] = (3 * s[i] - 2 * y_prime[i] - y_prime[i+1]) / h[i]
        c[i] = y_prime[i]
        d[i] = cdf[i]
    end

    # Create interpolation function
    itp(x) = begin
        @assert data[1] <= x <= data[end]
        i = searchsortedlast(data, x)
        i = min(i, n - 1)
        i = max(i, 1)

        dx = x - data[i]
        return a[i] * dx^3 + b[i] * dx^2 + c[i] * dx + d[i]
    end

    itp_gradient = x -> begin
        @assert data[1] <= x <= data[end]
        i = searchsortedlast(data, x)
        i = min(i, n - 1)
        i = max(i, 1)

        dx = x - data[i]
        return 3 * a[i] * dx^2 + 2 * b[i] * dx + c[i]
    end

    itp, itp_gradient
end

function steffen_monotonic_interpolation_bootstrap(data::AbstractVector{<:Real}; bootstrap_frac::Float64=0.01, num_samples::Int=50)
    @assert 0 < bootstrap_frac <= 1
    @assert num_samples > 0

    n = length(data)
    data = sort(data)
    itps = []
    itp_gradients = []
    for _ in 1:num_samples
        idx = sample(2:n-1, floor(Int, n * bootstrap_frac), replace=true)
        bootstrap_data = append!([data[1]], data[idx], [data[end]])
        itp, itp_gradient = steffen_monotonic_interpolation(bootstrap_data)
        push!(itps, itp)
        push!(itp_gradients, itp_gradient)
    end

    itp(x) = mean([itp(x) for itp in itps])
    itp_gradient(x) = mean([itp_gradient(x) for itp_gradient in itp_gradients])
    itp, itp_gradient
end

function interpolate_data(data::AbstractVector{<:Real}, interp_name::String)
    """
        Interpolate the data using the specified interpolation method.
        We estimate the cdf of the data, interpolate that, then take the derivative.
    """

    return steffen_monotonic_interpolation_bootstrap(data)
end

function plot_bandwidths(interp_name)
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
    cfdm = central_fdm(20, 1; factor=1e3)
    for (name, dist) in distributions
        x = [eval(dist) for _ in 1:n]
        interp, interp_gradient = interpolate_data(x, interp_name)

        x_min = minimum(x)
        x_max = maximum(x)
        eval_x = collect(x_min:std(x)/100:x_max)
        eval_y = interp_gradient.(eval_x)

        ppdf = plot(x=eval_x, y=eval_y, Geom.line, Guide.xlabel("X"), Guide.ylabel("Density"), Guide.title(name))
        pcdf = plot(x=eval_x, y=interp.(eval_x), Geom.line, Guide.xlabel("X"), Guide.ylabel("CDF"), Guide.title(name))
        push!(pdf_plots, ppdf)
        push!(cdf_plots, pcdf)
    end

    dir = joinpath("plots", "cdf-interpolation")
    isdir(dir) || mkpath(dir)
    for (name, plots) in [("pdf", pdf_plots), ("cdf", cdf_plots)]
        stack = gridstack(convert(Matrix{Plot}, reshape(plots, 2, 2)))
        save_plot(stack, joinpath(dir, "$(interp_name)-$(name).pdf"))
    end
end

for (name, interp) in interpolations
    plot_bandwidths(name)
end