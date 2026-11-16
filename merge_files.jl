include("reconstruct_dataset.jl")

nsh = 4

for case = 1:10
    prefix = "Testing/c$(case)/"
    files = String[]
    for sh = 0:nsh-1
        file = "full_recon_s$(sh)of$(nsh).nc"
        push!(files,prefix*file)
    end
    outfile = prefix*"full_recon.nc"
    merge_shards(files, outfile)
end