struct MeanAccuracyEstimator <: AccuracyEstimator
end

function accuracy(_::MeanAccuracyEstimator, data::AbstractVector{T})::T where T
    return mean(data)
end
