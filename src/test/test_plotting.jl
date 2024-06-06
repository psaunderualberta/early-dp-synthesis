
"""
Test cases for ./test/test_plotting.jl
"""

using Test
import Gadfly: Plot
import Distributions: Normal
import FromFile: @from
@from "../util/plotting.jl" import kde_density_plot, save_plot

function test_plotting()
    @testset "KDE Density Plot Constant" begin
        # Test KDE Density Plot produces a Gadfly.Plot
        data = fill(1.0, 100)
        plot = kde_density_plot("1.0", data)
        @test isa(plot, Plot)
    end

    @testset "KDE Density Plot" begin
        # Test KDE Density Plot produces a Gadfly.Plot
        data = rand(Normal(0, 1), 100)
        plot = kde_density_plot("normal(0, 1)", data)
        @test isa(plot, Plot)
    end
end
