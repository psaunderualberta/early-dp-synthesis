"""
Test fitting kde

https://github.com/JuliaStats/KernelDensity.jl
"""

using KernelDensity
using Distributions
using Gadfly
using Interpolations
using Cairo

function test_kde()
    n = 10000

    # Define distribution
    x = rand(Uniform(0, 1), n)

    fn = kde(x; boundary=(minimum(x), maximum(x)))
    ik = InterpKDE(fn)

    # Check that the pdf is positive for all x
    for _ in 1:n
        t = rand(Uniform(minimum(x), maximum(x)))
        @assert pdf(ik, t) > 0 "$(t), $(minimum(x)), $(maximum(x))"
    end

    # Plot the histogram
    p = plot(x=x, Geom.histogram(bincount=1000, density=true), Guide.xlabel("x"), Guide.ylabel("Density"),
        layer(x -> pdf(ik, x),minimum(x) - std(x), maximum(x) + std(x), color=[colorant"black"]),
    )

    # Draw the svg
    img = PDF("kde.pdf", 6inch, 4inch)
    draw(img, p)

end

test_kde()
