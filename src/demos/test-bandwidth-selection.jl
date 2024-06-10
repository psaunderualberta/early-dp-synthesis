using Gadfly
using StatsBase
import FromFile: @from

@from "../util/distributions.jl" import uniform, normal, laplace
@from "../util/plotting.jl" import save_plot

struct KDE
    x_i::AbstractVector{<:Real}
    h::Union{Real,Nothing}
    kernel::Function
    kernel_derivative::Function
    kernel_second_derivative::Function
end

const kernels = Dict(
    "gaussian" => (x, x_i, h, n) -> 1 / sqrt(2 * pi) * exp(-1 / 2 * (x - x_i)^2 / h^2),
)

const kernel_derivatives = Dict(
    "gaussian" => (x, x_i, h, n) -> (x - x_i)^2 * exp(-1 / 2(x - x_i)^2 / h^2) / (h^2 * n * sqrt(2 * pi)) * (1 / h^2 - 1)
)

const kernel_second_derivatives = Dict(
    "gaussian" => (x, x_i, h, n) -> exp(-1 / 2 * (x - x_i)^2 / h^2) * (2 * h^4 - 5 * h^2 * (x - x_i)^2 + (x - x_i)^4) / (h^7 * n * sqrt(2 * pi))
)

function KDE(data::AbstractVector{<:Real}, h::Real; k::String="gaussian")
    """
        Create a KDE object from the data and bandwidth.
    """
    kernel = kernels[k]
    kernel_derivative = kernel_derivatives[k]
    kernel_second_derivative = kernel_second_derivatives[k]

    bins = fit(Histogram, data).edges[1]

    return KDE(bins, h, kernel, kernel_derivative, kernel_second_derivative)
end

function H(kde::KDE, test_hist::Histogram, h::Real)
    """
        Compute 1 / 2 * \\sum_{i=1}^{n} (f_x - fhat_h(x))^2.
        This denotes how well the kernel density estimate fits the data.
    """

    n = length(test_hist.weights)

    test_weights = test_hist.weights
    test_edges = collect(test_hist.edges[1][1:end-1])

    return 1 / 2 * sum([(test_weights[i] - fhat_h(kde, test_edges[i], h))^2 for i in 1:n])
end

function H_prime(kde::KDE, test_hist::Histogram, h::Real)
    """
        Compute the derivative of H with respect to h.
    """

    n = length(test_hist.weights)

    test_weights = test_hist.weights
    test_edges = test_hist.edges[1:end-1]

    return sum([
        (fhat_h(kde, test_edges[i], h) - test_weights[i]) * fhat_h_prime(kde, test_edges[i], h)
        for i in 1:n
    ])

end

function H_prime_prime(kde::KDE, test_hist::Histogram, h::Real)
    """
        Compute the second derivative of H with respect to h.
    """

    n = length(test_hist.weights)

    test_weights = test_hist.weights
    test_edges = test_hist.edges[1:end-1]

    return sum([
        fhat_h_prime(kde, test_edges[i], h)^2 + (fhat_h(kde, x, h) - test_weights[i]) * fhat_h_prime_prime(kde, test_edges[i], h)
        for i in 1:n
    ])
end

function fhat_h(kde::KDE, x::Real, h::Real)
    """
        Compute the kernel density estimate of x given the data xs and bandwidth h.
    """

    @assert h > 0 "Bandwidth must be positive"

    return sum([kde.kernel(x, xi, h, length(kde.x_i)) for xi in kde.x_i])
end

function fhat_h_prime(kde, x::Real, h::Real)
    """
        Compute the derivative of the kernel density estimate of x given the data xs and bandwidth h.
    """

    @assert h > 0 "Bandwidth must be positive"

    return sum([kde.kernel_derivative(x, xi, h, length(kde.x_i)) for xi in kde.x_i])
end

function fhat_h_prime_prime(kde, x::Real, h::Real)
    """
        Compute the second derivative of the kernel density estimate of x given the data xs and bandwidth h.
    """

    @assert h > 0 "Bandwidth must be positive"

    n = length(kde.x_i)
    return sum([kde.kernel_second_derivative(x, xi, h, length(kde.x_i)) for xi in kde.x_i])
end


# function newton(x <: Real, x_i::AbstractVector{<:Real}, initial_h::Real; iters::Int=10)
#     """
#         Perform several iterations of newton's method to find a good estimate for the bandwidth.
#     """
#     h = initial_h
#     for _ in 1:iters
#         h = h - H_prime(x, x_i, h, "gaussian") / H_prime_prime(x, x_i, h, "gaussian")
#     end

#     h
# end

function plot_bandwidths()
    """
        Plot the kernel density estimate of the data x_i with bandwidth h.
    """

    n = 10_000
    hs = collect(0.05:0.01:20.0)

    sensitivity = 1.0
    distributions = Dict(
        "Uniform(0, 1)" => :(uniform(0, 1)),
        "Uniform(-5, 5)" => :(uniform(-5, 5)),
        "Normal(0, 1)" => :(normal(0, 1)),
        "Laplace(Unif(-5, 5), 1)" => :(laplace(uniform(-5, 5), 1)),
    )

    plots = []
    for (name, dist) in distributions
        x = [eval(dist) for _ in 1:n]

        test_hist = fit(Histogram, x, nbins=100)

        p = plot(x=hs, y=[H(KDE(x, h), test_hist, h) for h in hs], Geom.line, Guide.xlabel("Bandwidth"), Guide.ylabel("H"), Guide.title(name))
        push!(plots, p)
    end


    mat = convert(Matrix{Plot}, reshape(plots, 2, 2))
    return gridstack(mat)
end

function test_bandwidth_estimation()
    n = 10000

    sensitivity = 1.0
    distributions = Dict(
        "Laplace(Unif(-5, 5), 1)" => :(laplace(uniform(-5, 5), 1)),
        "Uniform(-5, 5)" => :(uniform(-5, 5)),
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
    end
end

p = plot_bandwidths()
save_plot(p, "bandwidths.pdf")