using KernelDensity
using Distributions

include("../misc/eval_tree.jl")

struct KDEPrivacyEstimator <: PrivacyEstimator
end


function varepsilon(est::KDEPrivacyEstimator, tree, dataset::Dataset{T,L}, options, prediction::AbstractVector{T}, params::Dict; num_iters::Int64=100)::L where {T,L}

    # function varepsilon(kde1::KDE, kde2::KDE, sampler::ContinuousUnivariateDistribution, num_iters::Int64=1000)::Float64 where {T}

    if haskey(params, "use_predictions") && params["use_predictions"]
        prediction = eval(tree, dataset, options)
    end

    # Get original distributions
    limits = (minimum(prediction), maximum(prediction))
    model = kde(prediction; boundary=limits)
    model_ik = InterpKDE(model)

    # Dist of data when shifted by maximum possible amount.
    varnames = dataset.variable_names
    sensitivity_col_idx = findfirst(item -> item == SENSITIVITY_COLUMN_NAME, varnames)
    sensitivity = dataset.X[sensitivity_col_idx, 1]  # The column is the same value
    sens_model = kde(prediction .- sensitivity)
    sens_model_ik = InterpKDE(sens_model)

    # Get the method of sampling
    resolution = 1000

    # Totally arbitrary choice for min & max of sampling
    # TODO: Since we're using kdes, we should probably make use of a distribution other than the gaussian
    vareps = L(0)

    # Use random sampling to estimate varepsilon
    for x in limits[1]:1/resolution:limits[2]
        model_prob = pdf(model_ik, x) + 1e-5
        @assert model_prob > 0 "$(string_tree(tree, options)), $x, $sampler, $prediction"

        # If this datapoint is impossible under neighbouring distributions,
        # then the privacy parameter (varepsilon) is infinite
        sens_model_prob = pdf(sens_model_ik, x)
        if sens_model_prob == 0
            return L(Inf)
        end

        diff = model_prob / sens_model_prob
        vareps = max(vareps, max(diff, 1 / diff))
    end

    return L(log(vareps))
end