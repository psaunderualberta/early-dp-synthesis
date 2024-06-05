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

    # Get original distributions
    limits = (minimum(prediction) - sensitivity, maximum(prediction) + sensitivity)
    model = kde(prediction; boundary=limits)
    model_ik = InterpKDE(model)

    # Get the distribution when data is shifted left by sensitivity
    sens_model = kde(prediction .- sensitivity; boundary=limits .- sensitivity)
    sens_model_ik = InterpKDE(sens_model)

    # Totally arbitrary choice for min & max of sampling
    # TODO: Since we're using kdes, we should probably make use of a distribution other than the gaussian
    vareps = L(0)

    # Use random sampling to estimate varepsilon
    samples, complete = eval_tree_array(tree, dataset.X, options)
    if !complete
        return L(Inf)
    end

    # @assert complete "Evaluation of the tree '$(string_tree(tree, options))' failed. This should've been caught earlier."
    for x in samples
        model_prob = pdf(model_ik, x)

        # This datapoint being impossible is not an issue, since our bounds are 
        # outside the min & max of the sampled data itself
        if model_prob <= 0
            continue
        end

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

### Non Privacy Estimator

struct NonPrivacyEstimator <: PrivacyEstimator
end

function varepsilon(_::NonPrivacyEstimator, tree, dataset::Dataset{T,L}, options, prediction::AbstractVector{T}, params::Dict)::L where {T, L}
    return L(0.0)
end