using Distributions
using StatsBase
using LinearAlgebra


function test()
    unif = rand(Uniform(0, 1), 10000)
    normal = rand(Normal(0, 1), 10000)
    laplace = rand(Laplace(0, 1), 10000)

    unifh = fit(Histogram, unif, nbins=1000)
    normalh = fit(Histogram, normal, nbins=1000)
    laplaceh = fit(Histogram, laplace, nbins=1000)

    unifh = normalize(unifh, mode=:pdf)
    normalh = normalize(normalh, mode=:pdf)
    laplaceh = normalize(laplaceh, mode=:pdf)

    println(unifh)
    println(normalh)
    println(laplaceh)

end

test()