struct UnbalancedAccuracyEstimator <: AccuracyEstimator
end

function accuracy(_::UnbalancedAccuracyEstimator, data::AbstractVector{T})::T where T
    """
    	unbalanced_dist(tree, dataset, options)
    
    A loss function that aims to maximize the discrepancy between the number of elements to the left & right of the mean.
    """

    mn = mean(data)
    ltm = length(data[data.<=mn])
    gtm = length(data[data.>=mn])

    # 1e-5 to account for possible division by 0
    return length(data) / (abs(ltm - gtm) + 1e-5)
end