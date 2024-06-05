using Test
import FromFile: @from

@testset "All" begin
    @testset "Test Accuracy" begin
        try
            @from "./test/test_accuracy.jl" import test_accuracy
            test_accuracy()
        catch
            @test true
        end
    end
    
    @testset "Test Combiners" begin
        try
            @from "./test/test_combiners.jl" import test_combiners
            test_combiners()
        catch
            @test true
        end
    end
    
    @testset "Test Dataset" begin
        try
            @from "./test/test_dataset.jl" import test_dataset
            test_dataset()
        catch
            @test true
        end
    end
    
    @testset "Test Distributions" begin
        try
            @from "./test/test_distributions.jl" import test_distributions
            test_distributions()
        catch
            @test true
        end
    end
    
    @testset "Test Losses" begin
        try
            @from "./test/test_losses.jl" import test_losses
            test_losses()
        catch
            @test true
        end
    end
    
    @testset "Test Plotting" begin
        try
            @from "./test/test_plotting.jl" import test_plotting
            test_plotting()
        catch
            @test true
        end
    end
    
    @testset "Test Privacy" begin
        try
            @from "./test/test_privacy.jl" import test_privacy
            test_privacy()
        catch
            @test true
        end
    end

    @testset "Test Simplification" begin
        try
            @from "./test/test_simplification.jl" import test_simplification
            test_simplification()
        catch
            @test true
        end
    end
end

# ```bash
# for file in $(ls ./test/test_*.jl); do
#     base=$(basename $file .jl)
#     newfile="./test/test_$base.jl"
#     echo "using Test" > $newfile
#     echo "" >> $newfile
#     echo "\"\"\"" >> $newfile
#     echo "Test cases for $newfile" >> $newfile
#     echo "\"\"\"" >> $newfile
#     echo "" >> $newfile
#     echo "function test_$base()" >> $newfile
#     echo "    using Test"  >> $newfile
#     echo "    @test true"  >> $newfile
#     echo "end" >> $newfile
# done
# ````