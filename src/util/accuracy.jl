"""
This file contains the loss functions used to evaluate the accuracy of a distribution.
"""

import FromFile: @from
@from "../Constants.jl" import AccuracyEstimator

using Distributions: mean
 
### Linear mean loss function

struct LinearMeanAccuracyEstimator <: AccuracyEstimator
end

function accuracy(_::LinearMeanAccuracyEstimator, data::AbstractVector{T})::T where T
    return abs(mean(data))
end

### Quadratic mean loss function

struct QuadraticMeanAccuracyEstimator <: AccuracyEstimator
end

function accuracy(_::QuadraticMeanAccuracyEstimator, data::AbstractVector{T})::T where T
    return mean(data) ^ 2
end

### Linear loss function

struct LinearAccuracyEstimator <: AccuracyEstimator
end

function accuracy(_::LinearAccuracyEstimator, data::AbstractVector{T})::T where T
    return sum(abs.(data))
end

### Quadratic loss function

struct QuadraticAccuracyEstimator <: AccuracyEstimator
end

function accuracy(_::QuadraticAccuracyEstimator, data::AbstractVector{T})::T where T
    return sum(data .^ 2)
end

### Unbalanced loss function

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
