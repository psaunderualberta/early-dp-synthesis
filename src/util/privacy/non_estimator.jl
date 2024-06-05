import FromFile: @from
@from "../../Constants.jl" import PrivacyEstimator

struct NonPrivacyEstimator <: PrivacyEstimator
end

function varepsilon(_::NonPrivacyEstimator, tree, dataset::Dataset{T,L}, options, prediction::AbstractVector{T}, params::Dict)::L where {T, L}
    return L(0.0)
end