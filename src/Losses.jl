module Losses

export zero_mean_dist,
    unbalanced_dist,
    privacy_loss

using SymbolicRegression: Dataset, eval_tree_array
using Distributions
using KernelDensity

include("Common.jl")
using .Common

# Type alias for KDEs
KDE = Union{UnivariateKDE,BivariateKDE}

# TODO: Perhaps make use of optuna later



function privacy_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
	# TODO: Possibly use different samples to estimate privacy vs. accuracy,
	# or the different kdes?

    prediction, flag = eval_tree_array(tree, dataset.X, options)

    # Either evaluation failed, or the distribution is approximately a constant
    if !flag || std(prediction) < 1e-5
        return L(Inf)
    end



	# Estimate the privacy loss
	varepsilon = estimate_varepsilon_kde(kde, sens_kde, sampler)

	# Get the accuracy
	accuracy = mean(prediction)

	# Combine privacy & accuracy
end


"""
TEST LOSS FUNCTIONS
"""
function zero_mean_dist(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    """
    	zero_mean_dist(tree, dataset, options)

    A loss function for whether the distribution induced by the expression 'tree' is centred around 0.
    The loss is simply the absolute value of the distribution's mean. 

    Note that, if the distribution has a standard deviation of approximately 0 ($leq 10^{-5}), then the loss is $infty.
    """
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag || std(prediction) < 1e-5
        return L(Inf)
    end

    return abs(mean(prediction))
end

end