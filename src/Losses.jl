using SymbolicRegression: Dataset, eval_tree_array

include("Common.jl")

# TODO: Perhaps make use of optuna later

function privacy_loss(accest, privest, combest, tree, dataset::Dataset{T,L}, options)::L where {T,L}
    predictions, flag = eval_tree_array(tree, dataset.X, options)

    # Additional constraints
    flag = flag && std(predictions) > 1e-5
    if !flag
        if !isnan(std(predictions))
            println(string_tree(tree, options), " | ", predictions[1:3], " | ", std(predictions))
        end
        return L(Inf)
    end

    # TODO: Replace dicts with args passed in
    acc = accuracy(accest, predictions)
    vareps = varepsilon(privest, tree, dataset, options, predictions, Dict())
    loss = combine(combest, acc, vareps, Dict("lambda" => 5))

    is = any(isnan.(predictions))
    @assert !isnan(loss) "$is, $acc, $vareps, $loss"
    return loss
end

"""
WTFFFFFFFF
true | (-1.3776619517892104 - -1.0812064844633116) + unif(-0.5559419042937783, -0.3831453089540088) | [0.17912695754478403, 0.17912695754478403, 0.17912695754478403] | 8.327089049550533e-17
"""
function lf(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    n = size(dataset.X, 2)
    predictions = Vector{L}(undef, n)
    incomplete = false

    for i in 1:n
        pred, complete = eval_tree_array(tree, dataset.X[:, i:i], options)
        @assert length(pred) == 1
        predictions[i] = pred[1]
        incomplete = incomplete || !complete

        if incomplete
            break
        end
    end

    # Additional constraints
    if incomplete || std(predictions) <= 1e-5
        # if !incomplete && !isnan(std(predictions))
        #     st = string_tree(tree, options)
        #     weird = (contains(st, "unif") || contains(st, "normal")) && length(unique(predictions)) == 1
        #     if weird
        #         println()
        #         println(complete, " | ", string_tree(tree, options), " | ",
        #             [eval_tree_array(tree, dataset.X[:, i:i], options)[1] for i in 1:3],
        #             " | ", std(predictions)
        #         )
        #     end
        # end
        return L(typemax(Int64))
    end


    abs(mean(predictions))
end
