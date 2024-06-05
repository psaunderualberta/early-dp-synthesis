"""
This file contains the loss functions used to combine the accuracy and privacy loss of a distribution.
"""

import FromFile: @from
@from "../Constants.jl" import Combiner

### Linear Combiner

struct LinearCombiner <: Combiner
end

function combine(_::LinearCombiner, accuracy::Float64, epsilon::Float64, params::Dict)::Float64
    if !haskey(params, "lambda")
        error("Linear Combinator does not have required parameter 'lambda'!")
    end 

    return abs(accuracy + epsilon * params["lambda"])
end

### Product Combiner

struct ProductCombiner <: Combiner
end

function combine(_::ProductCombiner, accuracy::Float64, epsilon::Float64, params::Dict)::Float64
    return abs(accuracy * epsilon)
end
