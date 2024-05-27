using SymbolicRegression: Dataset, eval_tree_array


function eval(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    distribution, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end

    return distribution
end
