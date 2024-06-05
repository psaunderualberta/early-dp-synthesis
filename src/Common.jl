import FromFile: @from

# Import all the necessary types and functions
@from "./util/accuracy.jl" import LinearAccuracyEstimator,
    QuadraticAccuracyEstimator,
    LinearMeanAccuracyEstimator,
    QuadraticMeanAccuracyEstimator,
    UnbalancedAccuracyEstimator
@from "./util/privacy.jl" import KDEPrivacyEstimator, NonPrivacyEstimator
@from "./util/combiners.jl" import LinearCombiner, ProductCombiner


# Publicly exposed estimators
const accuracy_estimators = Dict(
    "linear" => LinearAccuracyEstimator,
    "quadratic" => QuadraticAccuracyEstimator,
    "linear-mean" => LinearMeanAccuracyEstimator,
    "quadratic-mean" => QuadraticMeanAccuracyEstimator,
    "unbalanced" => UnbalancedAccuracyEstimator
)

const privacy_estimators = Dict(
    "kde" => KDEPrivacyEstimator,
    "none" => NonPrivacyEstimator
)

const combiners = Dict(
    "linear" => LinearCombiner,
    "product" => ProductCombiner
)
