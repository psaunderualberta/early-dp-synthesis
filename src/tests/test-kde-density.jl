"""
Test fitting kde

https://github.com/JuliaStats/KernelDensity.jl
"""

using KernelDensity
using Distributions
using Gadfly
using Interpolations
using Cairo

include("../util/simplification/simplify.jl")

function unif(a::T, b::T) where {T}
    a < b ? rand(Uniform(a, b)) : T(NaN)
end

function laplace_unif(a, b)
    if a < b
        x = rand(Uniform(a, b))
        return rand(Laplace(x, 1))
    end

    return NaN
end

function laplace(b)
    if b > 0
        return rand(Laplace(0, b))
    end

    return NaN
end

function normal(a, b)
    if b > 0
        return rand(Normal(a, b))
    end

    return NaN
end


function test_kde()
    n = 10000

    sensitivity = 1.0
    distributions = Dict(
        "Laplace(Unif(-5, 5), 1)" => :(laplace_unif(-5, 5)),
        "Uniform(-5, 5)" => :(unif(-5, 5)),
        "Initial-Early-DP-Synth-Result" => :(normal(
                laplace(10.113704306677194),
                normal(
                    ((10.113704306677194 + (10.113704306677194 + 10.113704306677194)) + (10.113704306677194 + (10.113704306677194 + (10.113704306677194 + 10.113704306677194)))) - sensitivity,
                    sensitivity
                ))),
    )


    for (name, dist) in distributions
        x = [eval(dist) for _ in 1:n]

        fn = kde(x; boundary=(minimum(x), maximum(x)))
        ik = InterpKDE(fn)

        # Check that the pdf is positive for all x
        for _ in 1:n
            t = rand(Uniform(minimum(x), maximum(x)))
            @assert pdf(ik, t) > 0 "$(t), $(minimum(x)), $(maximum(x))"
        end

        # Plot the histogram
        title = simplify_numeric(dist)
        p = plot(x=x, Geom.histogram(bincount=1000, density=true), color=[colorant"orange"],
            Guide.xlabel("x"), Guide.ylabel("Density"), Guide.title("$(title)"),
            layer(x -> pdf(ik, x), minimum(x) - std(x), maximum(x) + std(x), color=[colorant"black"]),
        )

        # Draw the svg
        img = PDF("$(name).pdf", 6inch, 4inch)
        draw(img, p)
    end

end

test_kde()
