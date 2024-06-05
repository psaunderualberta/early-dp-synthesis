import FromFile: @from

# Import all the necessary types and functions
@from "./util/accuracy/linear.jl" import LinearAccuracyEstimator
@from "./util/accuracy/quadratic.jl" import QuadraticAccuracyEstimator
@from "./util/privacy/kde-estimator.jl" import KDEPrivacyEstimator
@from "./util/privacy/non-estimator.jl" import NonPrivacyEstimator
@from "./util/combiners/linear.jl" import LinearCombiner
@from "./util/combiners/product.jl" import ProductCombiner


# Publicly exposed estimators
const accuracy_estimators = Dict(
    "linear" => LinearAccuracyEstimator,
    "quadratic" => QuadraticAccuracyEstimator
)

const privacy_estimators = Dict(
    "kde" => KDEPrivacyEstimator,
    "none" => NonPrivacyEstimator
)

const combiners = Dict(
    "linear" => LinearCombiner,
    "product" => ProductCombiner
)
