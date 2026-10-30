using Pkg
Pkg.activate("docs")
Pkg.develop(Pkg.PackageSpec(path="."))
using PlutoSliderServer

PlutoSliderServer.export_directory("examples")


## Copy files
using Glob

# Source and destination directories
src_dir = "examples"
dest_dir = "docs/src/public/pluto"  # Change this to your desired destination

# Create destination if it doesn't exist
mkpath(dest_dir)

# Copy all files from source to destination
for file in readdir(src_dir, join=true)
    if isfile(file)
        cp(file, joinpath(dest_dir, basename(file)), force=true)
        println("Copied: $(basename(file))")
    end
end

println("Done! All files copied from $(src_dir) to $(dest_dir)")