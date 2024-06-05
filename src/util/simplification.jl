"""
Performs simplification of numeric algebraic expressions, represented as quotes.
Function calls are not simplified, but algebraic expressions are simplified.

i.e. `simplify_numeric(:(1 + 2))` returns `:(3)` `(3)`, but `simplify_numeric(:(f(1) + 2))` returns `:(f(1) + 2)`.
"""

using Test

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

    e
end


"""
Test cases for simplification.jl
"""

function test_simplification() 
    @testset "Test Simplify Numeric" begin
        @test simplify_numeric(:(1 + 2)) == 3
        @test simplify_numeric(:(1 + (2 + 3))) == 6
        @test simplify_numeric(:(1 + (2 * 3))) == 7
        @test simplify_numeric(:((1 + 2) * 3)) == 9
        @test simplify_numeric(:(1 + (2 + e))) == :(1 + (2 + e))  # TODO: This should be simplified.
        @test simplify_numeric(:((1 + 2) + e)) == :(3 + e)
        @test simplify_numeric(:(f(1 + 2))) == :(f(3))
        @test simplify_numeric(:(f(3) + 3)) == :(f(3) + 3)
        @test simplify_numeric(:(f(1 + 2) + 3)) == :(f(3) + 3)
    end

    @testset "Test Insert Variables" begin
        variables = Dict("x" => 1, "y" => 2)

        e = "x + y"
        @test insert_variables(e, variables) == "1 + 2"

        e = "x + y + x"
        @test insert_variables(e, variables) == "1 + 2 + 1"

        e = "z + y + x"
        @test insert_variables(e, variables) == "z + 2 + 1"

        e = "z + (y + x) + z"
        @test insert_variables(e, variables) == "z + (2 + 1) + z"
        
        e = "z+(y+x)+z"
        @test insert_variables(e, variables) == "z+(2+1)+z"
    end
end
