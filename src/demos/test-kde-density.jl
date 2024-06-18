"""
Test fitting kde

https://github.com/JuliaStats/KernelDensity.jl
"""

using KernelDensity
using Distributions
using Gadfly
using Interpolations
using Cairo
using FromFile: @from

@from "../util/distributions.jl" import normal, uniform, laplace
@from "../util/privacy.jl" import estimate_varepsilon_kde

function test_kde()
    n = 10000

    sensitivity = 1.0
    distributions = Dict(
        "Normal(0, 1)" => :(normal(0, 1)),
        "laplace(0.00016)" => :(laplace(0.00016)),
        "Uniform(-5, 5)" => :(uniform(-5, 5)),
    )

    for (name, dist) in distributions
        epsilons = []
        for _ in 1:10
            x = [eval(dist) for _ in 1:n]
            eval_x = [eval(dist) for _ in 1:n]

            eps = estimate_varepsilon_kde(x, eval_x, 1.0)
            push!(epsilons, round(eps, digits=2))
        end

        println(name)
        println(epsilons)
        println(mean(epsilons), " ", std(epsilons))
    end
end

test_kde()
