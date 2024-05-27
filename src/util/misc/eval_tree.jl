using SymbolicRegression: Dataset, eval_tree_array


function eval(tree, dataset::Dataset{T,L}, options)::Tuple{AbstractVector{L}, Bool} where {T,L}
    distribution, flag = eval_tree_array(tree, dataset.X, options)

    # Additional constraints
    flag = flag && std(distribution) > 1e-5

    return distribution, flag
end
