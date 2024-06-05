"""
This module contains functions for plotting distributions of data and kernel density estimates.
"""

module KDEDensityPlot

export kde_density_plot

using KernelDensity
using Distributions
using Gadfly
using Interpolations
using Cairo

function kde_density_plot(distribution::String, n::Integer; kwargs...)
    """
    Plot the kernel density estimate of a distribution.

    Args:
        distribution (String): The distribution to sample from.
        n (Integer): The number of samples to take.
    
    Kwargs:
        hist_color (Color): The color of the histogram.
        kde_color (Color): The color of the line denoting the kernel density estimate.
        xlabel (String): The x-axis label.
        ylabel (String): The y-axis label.
        title (String): The title of the plot.
    """
    x = [eval(dist) for _ in 1:n]

    fn = kde(x; boundary=(minimum(x), maximum(x)))
    ik = InterpKDE(fn)

    # Check that the pdf is positive for all x
    for _ in 1:n
        t = rand(Uniform(minimum(x), maximum(x)))
        @assert pdf(ik, t) > 0 "$(t), $(minimum(x)), $(maximum(x))"
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
        layer(x -> pdf(ik, x), minimum(x) - std(x), maximum(x) + std(x), color=[kde_color]),
    )
end

end