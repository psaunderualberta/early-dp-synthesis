module Losses

export privacy_loss

using SymbolicRegression: Dataset, eval_tree_array

include("Common.jl")
using .Common

# TODO: Perhaps make use of optuna later

function privacy_loss(accest::AccuracyEstimator, privest::PrivacyEstimator, combest::Combiner, tree, dataset::Dataset{T,L}, options)::L where {T,L}
    predictions = eval(tree, dataset, options)

    acc = accuracy(accest, predictions)
    vareps = varepsilon(privest, tree, dataset, options, predictions, Dict())
    loss = combine(combest, acc, vareps, Dict("lambda" => 5))

    return loss
end

end
