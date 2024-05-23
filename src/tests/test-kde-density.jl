"""
Test fitting kde

https://github.com/JuliaStats/KernelDensity.jl
"""

using KernelDensity
using Distributions

function test_kde()
    x = rand(Normal(0, 1), 1000)
    
    fn = kde(x)
    println(pdf(fn, x))
    println(pdf(fn, 0))
end

test_kde()
