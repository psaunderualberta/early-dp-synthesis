module Losses

export zero_mean_dist,
	unbalanced_dist

using SymbolicRegression: Dataset, eval_tree_array
using Distributions

const NITER = 100

function get_accuracy_distribution(tree, dataset::Dataset{T,L}, options)::Float64 where {T,L}
	distribution, flag = eval_tree_array(tree, dataset.X, options)
	if !flag 
		return Float64(Inf)
	end
	return mean(distribution)
end


function zero_mean_dist(tree, dataset::Dataset{T,L}, options)::L where {T,L}
	"""
		zero_mean_dist(tree, dataset, options)

	A loss function for whether the distribution induced by the expression 'tree' is centred around 0.
	The loss is simply the absolute value of the distribution's mean. 

	Note that, if the distribution has a standard deviation of approximately 0 ($leq 10^{-5}), then the loss is $infty.
	"""
	prediction, flag = eval_tree_array(tree, dataset.X, options)
	if !flag || std(prediction) < 1e-5
		return L(Inf)
	end

	return abs(mean(prediction))
end

function unbalanced_dist(tree, dataset::Dataset{T,L}, options)::L where {T,L}
	"""
		unbalanced_dist(tree, dataset, options)
	
	A loss function that aims to maximize the discrepancy between the number of elements to the left & right of the mean.
	"""
	prediction, flag = eval_tree_array(tree, dataset.X, options)
	if !flag || std(prediction) < 1e-5
		return L(Inf)
	end

	mn = mean(prediction)
	ltm = length(prediction[prediction .<= mn])
	gtm = length(prediction[prediction .>= mn])

	# 1e-5 to account for possible division by 0
	return length(prediction) / (abs(ltm - gtm) + 1e-5)
end

end