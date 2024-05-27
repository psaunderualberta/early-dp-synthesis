struct ProductCombiner <: Combiner
end

function combine(_::ProductCombiner, accuracy::Float64, epsilon::Float64, params::Dict)::Float64
    return accuracy * epsilon
end
