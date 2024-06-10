"""
This module contains functions for plotting distributions of data and kernel density estimates.
"""

import FromFile: @from
@from "./simplification.jl" import simplify_numeric, insert_variables
@from "./distributions.jl" import uniform, normal, laplace

using KernelDensity
using Distributions
using Gadfly
using Compose
using Interpolations
using Cairo

### Functionality to plot a kernel density estimate of a distribution.

function kde_density_plot(distribution::String, data::AbstractVector{<:Real}; kwargs...)
    """
    Plot the kernel density estimate of a distribution.

    Args:
        distribution (String): The distribution to sample from.
        data (AbstractVector{Real}): The data to fit the kernel density estimate to.
    
    Kwargs:
        hist_color (Color): The color of the histogram.
        kde_color (Color): The color of the line denoting the kernel density estimate.
        xlabel (String): The x-axis label.
        ylabel (String): The y-axis label.
        title (String): The title of the plot.
    """

    # Get plot attributes from kwargs, or set to default
    hist_color = get(kwargs, :hist_color, colorant"blue")
    kde_color = get(kwargs, :kde_color, colorant"black")
    xlabel = get(kwargs, :xlabel, "x")
    ylabel = get(kwargs, :ylabel, "Density")
    title = get(kwargs, :title, simplify_numeric(Meta.parse(distribution)))

    limits = (minimum(data), maximum(data))
    if limits[1] == limits[2]
        return plot(x=data, Geom.histogram(bincount=1000, density=true), color=[hist_color],
            Guide.xlabel(xlabel), Guide.ylabel(ylabel), Guide.title("$(title)"),
        )
    end

    ik = InterpKDE(kde(data; boundary=limits))

    # Check that the pdf is nonnegative for all x, expanded slightly to account for constant distributions
    limits = (limits[1] - std(data) - 0.1, limits[2] + std(data) + 0.1)
    for _ in eachindex(data)
        t = rand(Uniform(limits...))
        @assert pdf(ik, t) >= 0 "$(t), $(limits[1]), $(limits[2])"
    end

    # Return the plot the histogram
    return plot(x=data, Geom.histogram(bincount=1000, density=true), color=[hist_color],
        Guide.xlabel(xlabel), Guide.ylabel(ylabel), Guide.title("$(title)"),
        layer(x -> pdf(ik, x), limits[1], limits[2], color=[kde_color]),
    )
end

### Functionality to save a plot to a file.

function save_plot(plot::Union{Plot, Context}, path::String)
    extensions = Dict(
        "pdf" => PDF,
        "png" => PNG,
        "svg" => SVG,
        "ps" => PS,
        "tex" => PGF,
    )

    # Get extension from filename
    extension = split(path, ".")[end]

    # Get extension type
    if haskey(extensions, extension)
        graphic_fun = extensions[extension]
        extension = draw(graphic_fun(path, 6inch, 4inch), plot)
    else
        error("Extension not supported: $extension")
    end
end

### Dictionary of registered valid plotting functions
const PLOTTING_FUNCTIONS = Dict(
    "kde_density_plot" => kde_density_plot,
)
