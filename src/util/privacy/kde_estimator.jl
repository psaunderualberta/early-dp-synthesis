using KernelDensity
using Distributions

include("../misc/eval_tree.jl")

struct KDEPrivacyEstimator <: PrivacyEstimator
    use_predictions::Bool
end


function varepsilon(e::KDEPrivacyEstimator, tree, dataset::Dataset{T,L}, options, prediction::AbstractVector{T}, params::Dict)::L where {T,L}

    # function varepsilon(kde1::KDE, kde2::KDE, sampler::ContinuousUnivariateDistribution, num_iters::Int64=1000)::Float64 where {T}

    if !e.use_predictions
        prediction = eval(tree, dataset, options)
    end

    # Get original distributions
    kde = kde(prediction)

    # Dist of data when shifted by maximum possible amount.
    varnames = dataset.variable_names
    sensitivity_col_idx = findfirst(item -> item == SENSITIVITY_COLUMN_NAME, varnames)
    sensitivity = dataset.X[sensitivity_col_idx, 1]  # The column is the same value
    sens_kde = kde(prediction .- sensitivity)

    # Get the method of sampling
    sampler = Uniform(min(prediction), max(prediction))

    # Totally arbitrary choice for min & max of sampling
    # TODO: Since we're using kdes, we should probably make use of a distribution other than the gaussian
    vareps = Float64(Inf)

    # Use random sampling to estimate varepsilon
    for _ in 1:num_iters
        x = rand(sampler)

        diff = pdf(kde, x) - pdf(sens_kde, x)
        vareps = max(vareps, max(diff, 1 / diff))
    end

    return vareps
end