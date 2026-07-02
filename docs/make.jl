using EDUC_BiDiM_IRM
using Documenter

DocMeta.setdocmeta!(EDUC_BiDiM_IRM, :DocTestSetup, :(using EDUC_BiDiM_IRM); recursive=true)

makedocs(;
    modules=[EDUC_BiDiM_IRM],
    authors="aTrotier <a.trotier@gmail.com> and contributors",
    sitename="EDUC_BiDiM_IRM.jl",
    format=Documenter.HTML(;
        canonical="https://aTrotier.github.io/EDUC_BiDiM_IRM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/aTrotier/EDUC_BiDiM_IRM.jl",
    devbranch="main",
)
