"""
Tests that the installation of optuna is performing as intended

https://juliamanifolds.github.io/ManoptExamples.jl/stable/examples/HyperparameterOptimization/#Summary
"""

using PyCall

# Second argument is fall-back to installation from conda
optuna = pyimport_conda("optuna", "optuna")

function objective(trial::PyObject)
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ^ 2
end

function study()
    study = optuna.create_study()
    study.optimize(objective; n_trials = 100)

    println(study.best_params, study.best_value)
end

study()