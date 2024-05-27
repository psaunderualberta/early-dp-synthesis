struct ProductCombiner <: Combiner
end

function combine(_::ProductCombiner, accuracy::Float64, epsilon::Float64, params::Dict)::Float64
    return abs(accuracy * epsilon)
end
