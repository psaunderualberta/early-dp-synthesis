"""
Test fitting kde

https://github.com/JuliaStats/KernelDensity.jl
"""

using KernelDensity
using Distributions

function test_kde()
    x = rand(Normal(0, 1), 10000)

    fn = kde(x)
end

test_kde()
