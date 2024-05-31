"""
Test fitting kde

https://github.com/JuliaStats/KernelDensity.jl
"""

using KernelDensity
using Distributions

function test_kde()
    x = rand(Uniform(0, 1), 10000)

    fn = kde(x)
    println(minimum(x), " | ", maximum(x))
    println(fn)
end

test_kde()
