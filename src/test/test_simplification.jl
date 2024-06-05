using Test
using FromFile: @from
@from "../util/simplification.jl" import simplify_numeric, insert_variables

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