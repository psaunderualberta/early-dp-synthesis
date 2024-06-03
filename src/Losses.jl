using SymbolicRegression: Dataset, eval_tree_array
using DynamicExpressions

include("Common.jl")

# TODO: Perhaps make use of optuna later

function privacy_loss(accest, privest, combest, tree, dataset::Dataset{T,L}, options)::L where {T,L}
    predictions, flag = eval_tree_array(tree, dataset.X, options)

    # Additional constraints
    if !flag || std(predictions) <= 1e-5

        # Detection for strange stuff: namely, that a valid stochastic expression produces the same value
        if flag && isfinite(std(predictions))
            st = string_tree(tree, options)

            # TODO: revert from hardcoding the values of the arguments
            weird = any(dist -> contains(st, dist), ["unif", "normal", "laplace"]) && length(unique(predictions)) == 1
            if weird
                println(string_tree(tree, options), " | ", predictions[1:5], " | ", std(predictions))
                exit("HELP")
            end
        end

        return L(Inf)
    end

    # TODO: Replace dicts with args passed in
    acc = accuracy(accest, predictions)
    @assert isfinite(acc) "$(string_tree(tree, options)), $(predictions[1:3]), $acc"

    vareps = varepsilon(privest, tree, dataset, options, predictions, Dict())
    loss = combine(combest, acc, vareps, Dict("lambda" => 5))

    @assert isfinite(sum(predictions)) "$acc, $vareps, $loss"
    return loss
end

"""
This loss function is for testing strange observed behaviours during synthesis
"""
function test_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    predictions, complete = eval_tree_array(tree, dataset.X, options)

    # Additional constraints
    if !complete || std(predictions) <= 1e-5
        if complete && isfinite(std(predictions))
            st = string_tree(tree, options)
            weird = (contains(st, "unif") || contains(st, "normal")) && length(unique(predictions)) == 1
            if weird
                println(complete, " | ", string_tree(tree, options), " | ", predictions[1:3],
                    # [eval_tree_array(tree, dataset.X[:, i:i], options)[1] for i in 1:3],
                    " | ", std(predictions)
                )
            end
        end
        return L(typemax(Int64))
    end


    abs(mean(predictions))
end
