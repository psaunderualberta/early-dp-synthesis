struct QuadraticAccuracyEstimator <: AccuracyEstimator
end

function accuracy(_::QuadraticAccuracyEstimator, data::AbstractVector{T})::T where T
    return mean(data) ^ 2
end
