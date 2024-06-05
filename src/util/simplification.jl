"""
Performs simplification of numeric algebraic expressions, represented as quotes.
Function calls are not simplified, but algebraic expressions are simplified.

i.e. `simplify_numeric(:(1 + 2))` returns `:(3)` `(3)`, but `simplify_numeric(:(f(1) + 2))` returns `:(f(1) + 2)`.
"""

simplify_numeric(e) = error("not expected type: $(typeof(e))")
simplify_numeric(e::Symbol) = e
simplify_numeric(e::Number; digits::Integer=2) = round(e, digits=digits)  # Round to 2 decimal places

function simplify_numeric(e::Expr)
    if e.head == :call
        e.args[2] = simplify_numeric(e.args[2])

        if length(e.args) == 3
            e.args[3] = simplify_numeric(e.args[3])

            if e.args[2] isa Number && e.args[3] isa Number
                e = eval(e)
            end
        end
    end

    e
end

function insert_variables(e::String, variables::Dict)
    """
    Insert variables into an expression.
    """
    for (key, value) in variables
        e = replace(e, key => value)
    end
end
