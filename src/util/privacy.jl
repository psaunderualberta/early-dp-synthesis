"""
This file contains the loss functions used to evaluate the privacy loss of a distribution.
"""

import FromFile: @from
@from "../Constants.jl" import PrivacyEstimator, SENSITIVITY_COLUMN_NAME

using KernelDensity
using Distributions
import SymbolicRegression: Dataset, eval_tree_array

### KDE Privacy Estimator

struct KDEPrivacyEstimator <: PrivacyEstimator
end

function varepsilon(_::KDEPrivacyEstimator, tree, dataset::Dataset{T,L}, options, prediction::AbstractVector{T}, params::Dict)::L where {T,L}
    if haskey(params, "use_predictions") && params["use_predictions"]
        prediction, _ = eval_tree_array(tree, dataset.X, options)
    end

    # Dist of data when shifted by maximum possible amount.
    varnames = dataset.variable_names
    sensitivity_col_idx = findfirst(item -> item == SENSITIVITY_COLUMN_NAME, varnames)
    sensitivity = dataset.X[sensitivity_col_idx, 1]  # The column is the same value

    # Use random sampling to estimate varepsilon
    eval_samples, complete = eval_tree_array(tree, dataset.X, options)
    if !complete
        return L(Inf)
    end

    return L(estimate_varepsilon_kde(prediction, eval_samples, sensitivity))
end

function estimate_varepsilon_kde(prediction::AbstractVector{Float64}, eval_samples::AbstractVector{Float64}, sensitivity::Float64)::Float64
    # Get original distributions
    limits = (minimum(prediction) - 2 * sensitivity, maximum(prediction) + 2 * sensitivity)
    model = kde(prediction; boundary=limits)
    model_ik = InterpKDE(model)

    # Get the distributions when data is shifted by sensitivity
    sens_model_left = kde(prediction .- sensitivity; boundary=limits .- sensitivity)
    sens_model_right = kde(prediction .+ sensitivity; boundary=limits .+ sensitivity)
    sens_model_iks = [
        InterpKDE(sens_model_left),
        InterpKDE(sens_model_right)
    ]

    # Totally arbitrary choice for min & max of sampling
    # TODO: Since we're using kdes, we should probably make use of a distribution other than the gaussian
    vareps = 0

    # @assert complete "Evaluation of the tree '$(string_tree(tree, options))' failed. This should've been caught earlier."
    for x in eval_samples
        model_prob = pdf(model_ik, x)

        # # This datapoint being impossible is not an issue, since our bounds might be 
        # # outside the min & max of the sampled data itself
        # if model_prob <= 0
        #     continue
        # end

        # If this datapoint is impossible under neighbouring distributions,
        # then the privacy parameter (varepsilon) is infinite
        for sens_ik in sens_model_iks
            sens_prob = pdf(sens_ik, x)
            if sens_prob == 0
                return Inf
            end

            diff = model_prob / sens_prob
            vareps = max(vareps, max(diff, 1 / diff))
        end
    end

    println("here")
    return log(vareps)
end

### Non Privacy Estimator

struct NonPrivacyEstimator <: PrivacyEstimator
end

function varepsilon(_::NonPrivacyEstimator, tree, dataset::Dataset{T,L}, options, prediction::AbstractVector{T}, params::Dict)::L where {T, L}
    return L(0.0)
end