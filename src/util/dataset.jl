using FromFile: @from
@from "../Constants.jl" import SENSITIVITY_COLUMN_NAME

function create_dataset(n::Integer)
    @assert n > 0 "Number of samples must be positive!"
    d = Dict(
        SENSITIVITY_COLUMN_NAME => fill(1.0, n),
    )

    X = NamedTuple(((Symbol(key), value) for (key, value) in d))
    y = @. zeros(Float64, n)
    return (X, y)
end