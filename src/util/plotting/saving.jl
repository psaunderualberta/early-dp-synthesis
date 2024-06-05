"""
Functionality to save a plot to a file.
"""

using Gadfly
using Cairo

function save_plot(plot::Plot, path::String)
    extensions = Dict(
        "pdf" => PDF,
        "png" => PNG,
        "svg" => SVG,
        "ps" => PS,
        "eps" => EPS,
        "tex" => PGF,
    )

    # Get extension from filename
    extension = split(path, ".")[end]

    # Get extension type
    if haskey(extensions, extension)
        graphic_fun = extensions[extension]
        extension = draw(graphic_fun(path, 6inch, 4inch), plot)
    else
        error("Extension not supported: $extension")
    end
end

