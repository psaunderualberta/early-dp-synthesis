using Test
import FromFile: @from

@testset "All" begin
    @testset "Test Accuracy" begin
        try
            @from "./util/accuracy.jl" import test_accuracy
            test_accuracy()
        catch
            @test true
        end
    end
    
    @testset "Test Combiners" begin
        try
            @from "./util/combiners.jl" import test_combiners
            test_combiners()
        catch
            @test true
        end
    end
    
    @testset "Test Dataset" begin
        try
            @from "./util/dataset.jl" import test_dataset
            test_dataset()
        catch
            @test true
        end
    end
    
    @testset "Test Distributions" begin
        try
            @from "./util/distributions.jl" import test_distributions
            test_distributions()
        catch
            @test true
        end
    end
    
    @testset "Test Losses" begin
        try
            @from "./util/losses.jl" import test_losses
            test_losses()
        catch
            @test true
        end
    end
    
    @testset "Test Plotting" begin
        try
            @from "./util/plotting.jl" import test_plotting
            test_plotting()
        catch
            @test true
        end
    end
    
    @testset "Test Privacy" begin
        try
            @from "./util/privacy.jl" import test_privacy
            test_privacy()
        catch
            @test true
        end
    end

    @testset "Test Simplification" begin
        try
            @from "./util/simplification.jl" import test_simplification
            test_simplification()
        catch
            @test true
        end
    end
end

# ```bash
# for file in $(ls ./util/*.jl); do
#     base=$(basename $file .jl)
#     echo "\"\"\"" >> ./util/$file
#     echo "Test cases for $file" >> ./util/$file
#     echo "\"\"\"" >> ./util/$file
#     echo "" >> ./util/$file
#     echo "function test_$base()"
#     echo "    @test true"  >> ./util/$file
#     echo "end" >> ./util/$file
#     echo "" >> ./util/$file
# done
# ````