module Losses

export zero_mean_dist,
    unbalanced_dist,
    privacy_loss

using SymbolicRegression: Dataset, eval_tree_array
using Distributions
using KernelDensity

# Type alias for KDEs
KDE = Union{UnivariateKDE,BivariateKDE}

# TODO: Perhaps make use of optuna later

function get_accuracy_distribution(tree, dataset::Dataset{T,L}, options)::Float64 where {T,L}
    distribution, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return Float64(Inf)
    end
    return mean(distribution)
end


function estimate_varepsilon_kde(kde1::KDE, kde2::KDE, data::AbstractVector{T}, num_iters::Int64=1000)::Float64 where {T}
    """
    	estimate_varepsilon_kde(kde1, kde2, data, num_iters)

    	Estimates the privacy loss between two distributions using the KDEs 'kde1' and 'kde2'.
    	The privacy loss is estimated by sampling from the distributions and computing the maximum difference between the two distributions.

    	Parameters:
    	- kde1: The first KDE
    	- kde2: The second KDE
    	- data: The data used to train the KDEs
    	- num_iters: The number of iterations to sample from the distributions

    	Returns:
    	- The estimated privacy loss
    """
    # Totally arbitrary choice for min & max of sampling
    # TODO: Since we're using kdes, we should probably make use of a distribution other than the gaussian
    datastd = std(data)
    sample_min = min(data)
    sample_max = max(data)
    dist = Uniform(sample_min, sample_max)
    vareps = Float64(Inf)

    # Use random sampling to estimate varepsilon
    for _ in 1:num_iters
        x = rand(dist)

        diff = pdf(kde1, x) - pdf(kde2, x)
        vareps = max(vareps, max(diff, 1 / diff))
    end

    return vareps
end


function privacy_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    # Either evaluation failed, or the distribution is approximately a constant
    if !flag || std(prediction) < 1e-5
        return L(Inf)
    end
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

function unbalanced_dist(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    """
    	unbalanced_dist(tree, dataset, options)
    
    A loss function that aims to maximize the discrepancy between the number of elements to the left & right of the mean.
    """
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag || std(prediction) < 1e-5
        return L(Inf)
    end

    mn = mean(prediction)
    ltm = length(prediction[prediction.<=mn])
    gtm = length(prediction[prediction.>=mn])

    # 1e-5 to account for possible division by 0
    return length(prediction) / (abs(ltm - gtm) + 1e-5)
end

end