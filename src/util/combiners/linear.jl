struct LinearCombiner <: Combiner
end

function combine(_::LinearCombiner, accuracy::Float64, epsilon::Float64, params::Dict)::Float64
    if !haskey(params, "lambda")
        error("Linear Combinator does not have required parameter 'lambda'!")
    end 

    return abs(accuracy + epsilon * params["lambda"])
end