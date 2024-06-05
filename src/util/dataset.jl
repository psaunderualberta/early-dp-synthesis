using FromFile: @from
@from "../Constants.jl" import SENSITIVITY_COLUMN_NAME

function create_dataset(n::Integer)
    @assert n > 0 "Number of samples must be positive!"
    values = Dict(
        SENSITIVITY_COLUMN_NAME => 1.0,
    )

    X = NamedTuple(((Symbol(key), fill(value, n)) for (key, value) in values))
    y = @. zeros(Float64, n)
    return (X, y, values)
end
