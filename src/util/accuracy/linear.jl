import FromFile: @from
@from "../../Constants.jl" import AccuracyEstimator

struct LinearAccuracyEstimator <: AccuracyEstimator
end

function accuracy(_::LinearAccuracyEstimator, data::AbstractVector{T})::T where T
    return abs(mean(data))
end
