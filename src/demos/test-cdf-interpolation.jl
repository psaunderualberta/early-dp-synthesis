using Gadfly
using Interpolations
using FiniteDifferences
import FromFile: @from

@from "../util/distributions.jl" import uniform, normal, laplace
@from "../util/plotting.jl" import save_plot


const interpolations = Dict(
    "linear" => Gridded(Linear()),
    # "cubic" => Gridded(Cubic(Line(OnGrid()))),
    # "quadratic" => Gridded(Quadratic(Line(OnGrid()))),
    # "LinearMonotonic" => LinearMonotonicInterpolation(),
    # "SteffanMonotonic" => SteffenMonotonicInterpolation(),
    # "FiniteDifferenceMonotonic" => FiniteDifferenceMonotonicInterpolation(),
    # "FritschButlandMonotonic" => FritschButlandMonotonicInterpolation(),
)

function interpolate_data(data::AbstractVector{<:Real}, interp_name::String)
    """
        Interpolate the data using the specified interpolation method.
        We estimate the cdf of the data, interpolate that, then take the derivative.
    """

    data = sort(data)
    n = length(data)

    cdf = [i / n for i in 1:n]
    itp_cdf = extrapolate(
        interpolate((data,), cdf, interpolations[interp_name]),
        Flat()
    );
    return itp_cdf
end

function plot_bandwidths(interp_name)
    """
        Plot the kernel density estimate of the data x_i with bandwidth h.
    """

    n = 10_000

    distributions = Dict(
        "Uniform(0, 1)" => :(uniform(0, 1)),
        "Uniform(-5, 5)" => :(uniform(-5, 5)),
        "Normal(0, 0.000167)" => :(normal(0.0, 0.000167)),
        "Laplace(Unif(-5, 5), 1)" => :(laplace(uniform(-5, 5), 1)),
    )

    cdf_plots = []
    pdf_plots = []
    cfdm = central_fdm(20, 1; factor=1e3)
    for (name, dist) in distributions
        x = [eval(dist) for _ in 1:n]
        interp = interpolate_data(x, interp_name)

        x_min = minimum(x)
        x_max = maximum(x)
        eval_x = collect(x_min-1:0.1:x_max+1)
        eval_y = [cfdm(interp, x_i) for x_i in eval_x]

        ppdf = plot(x=eval_x, y=eval_y, Geom.line, Guide.xlabel("X"), Guide.ylabel("Density"), Guide.title(name))
        pcdf = plot(x=eval_x, y=interp.(eval_x), Geom.line, Guide.xlabel("X"), Guide.ylabel("CDF"), Guide.title(name))
        push!(pdf_plots, ppdf)
        push!(cdf_plots, pcdf)
    end

    pdfmat = convert(Matrix{Plot}, reshape(pdf_plots, 2, 2))
    cdfmat = convert(Matrix{Plot}, reshape(cdf_plots, 2, 2))
    pdfstack = gridstack(pdfmat)
    cdfstack = gridstack(cdfmat)

    dir = joinpath("plots", "interpolation")
    isdir(dir) || mkpath(dir)
    save_plot(pdfstack, joinpath(dir, "$(interp_name)-pdf.pdf"))
    save_plot(cdfstack, joinpath(dir, "$(interp_name)-cdf.pdf"))
end

for (name, interp) in interpolations
    plot_bandwidths(name)
end