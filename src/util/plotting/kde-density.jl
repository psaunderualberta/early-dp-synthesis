"""
This module contains functions for plotting distributions of data and kernel density estimates.
"""

import FromFile: @from
@from "../simplification/simplify.jl" import simplify_numeric

using KernelDensity
using Distributions
using Gadfly
using Interpolations
using Cairo

function kde_density_plot(distribution::String, data::AbstractVector{Real}; kwargs...)
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
    fn = kde(data; boundary=(minimum(data), maximum(data)))
    ik = InterpKDE(fn)

    # Check that the pdf is positive for all x
    for _ in 1:n
        t = rand(Uniform(minimum(data), maximum(data)))
        @assert pdf(ik, t) > 0 "$(t), $(minimum(data)), $(maximum(data))"
    end

    # Get plot attributes from kwargs, or set to default
    hist_color = get(kwargs, :hist_color, colorant"blue")
    kde_color = get(kwargs, :kde_color, colorant"black")
    xlabel = get(kwargs, :xlabel, "x")
    ylabel = get(kwargs, :ylabel, "Density")
    title = get(kwargs, :title, simplify_numeric(Meta.parse(distribution)))

    # Return the plot the histogram
    return plot(x=x, Geom.histogram(bincount=1000, density=true), color=[hist_color],
        Guide.xlabel(xlabel), Guide.ylabel(ylabel), Guide.title("$(title)"),
        layer(x -> pdf(ik, x), minimum(data) - std(data), maximum(data) + std(data), color=[kde_color]),
    )
end
