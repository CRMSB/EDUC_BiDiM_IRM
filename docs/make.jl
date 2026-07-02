using EDUC_BiDiM_IRM
using Documenter, DocumenterVitepress

DocMeta.setdocmeta!(EDUC_BiDiM_IRM, :DocTestSetup, :(using EDUC_BiDiM_IRM); recursive=true)

makedocs(;
    modules=[EDUC_BiDiM_IRM],
    authors="aTrotier <a.trotier@gmail.com> and contributors",
    sitename="EDUC_BiDiM_IRM.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/CRMSB/EDUC_BiDiM_IRM.jl",
        devbranch = "main", # or master, trunk, ...
        devurl = "dev",
    ),
    pages=[
        "Home" => "index.md",
        "Pluto example" => "pluto_example.md"
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/CRMSB/EDUC_BiDiM_IRM.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
