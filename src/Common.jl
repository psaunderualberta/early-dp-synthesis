using SymbolicRegression

include("Constants.jl")

abstract type PrivacyEstimator end
abstract type AccuracyEstimator end
abstract type Combiner end

# Include accuracy estimators, privacy estimators, and combiners
function recursive_include(path::String)
    """
    Just include everything for now :)
    """
    items = sort(readdir(path))

    for item = items
        newpath = joinpath(path, item)

        if isdir(newpath)
            recursive_include(newpath)
        elseif isfile(newpath) && endswith(item, ".jl")
            include(newpath)
        end
    end
end

recursive_include(joinpath(@__DIR__, "util"))

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
