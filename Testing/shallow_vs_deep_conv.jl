include("../reconstruct_dataset.jl")
include("../density_scatter.jl")
include("../wrf-utils.jl")

function select_quadrant(nbins, src, U, V)
    th = mod2pi(atan(V[src], U[src]))
    return Int(floor(nbins * th/(2pi)))
end

function detect_quadrant(th0, margin, src, U, V)
    # detects if winds at level src are in the 
    # range of angles (th0-margin, th0+margin)
    u = U[src]
    v = V[src]
    lx = hypot(u,v)
    u0 = cos(th0)
    v0 = sin(th0)
    ang = u*u0 + v*v0
    return (ang > cos(margin))
end

function subsample_safe(arr, n)
    na = length(arr)
    return arr[randperm(na)[1:min(n,na)]]
end

nt = size(wrf_ds[:U],4)
width = 6
tidxs = width+1:(2width):nt
tisq = -width .+ tidxs

#cf = load_ensemble_from_netcdf("Testing/c$(case)/full_recon.nc")

idxs_deep = Int64[]
idxs_shallow = Int64[]
olrs = Float64[]

ns = size(cf.spectra,1)

deep_conv_cutoff = 200.0
for i = 1:ns
    x, y, t = cf.xs[i], cf.ys[i], cf.ts[i]
    olr = wrf_ds[:olr][x,y,t]
    push!(olrs, olr)
    if olr .< deep_conv_cutoff
        push!(idxs_deep, i)
    else
        push!(idxs_shallow, i)
    end
end

# divide indices by wind direction at a plausible source level
src_guess = 6   # km, now in U_raw
nbins = 4
idx_bins_deep = [Int64[] for i = 1:nbins]
idx_bins_shallow = [Int64[] for i = 1:nbins]

function bin_by_wind!(bins, idxs, nbins, srclev)
    for j in idxs
        x, y, t = cf.xs[j], cf.ys[j], cf.ts[j]
        b = select_quadrant(nbins, srclev, wrf_ds[:U_raw][x,y,:,t], wrf_ds[:V_raw][x,y,:,t])
        push!(bins[b+1], j)
    end
end
bin_by_wind!(idx_bins_deep,    idxs_deep,    nbins, src_guess)
bin_by_wind!(idx_bins_shallow, idxs_shallow, nbins, src_guess)

#=
begin
# overlay best-fit line for olr vs flux
flux = dropdims(sum(cf.spectra[:,3:3:end],dims=2),dims=2)
mat = hcat(ones(length(olrs)), olrs)
coeff = mat\flux
olr_grid = 150.0:300.0
pred = coeff[1] .+ coeff[2].*olr_grid

#ds = density_scatter(olrs, 1e3*flux; radius=(1,0.5), n=(200,100), normalize=true, xlims=(200,300), ylims=(0,30))
ds = scatter(olrs, 1e3*flux, xlabel="OLR (W/m^2)", ylabel="Flux (mPa)", label="data")
plot!(ds, olr_grid, 1e3*pred, label="fit", linewidth=3)
savefig(ds, "Testing/c$(case)/olr_vs_flux.png")
end
=#

spec_deep = cf.spectra[idxs_deep,:]
spec_shallow = cf.spectra[idxs_shallow,:]

N = 2000
sub_idxs_deep = subsample_safe(idxs_deep,N)
sub_idxs_shal = subsample_safe(idxs_shallow,N)
csd = condense_spectra(cf.spectra[sub_idxs_deep,:], cf.nxs[sub_idxs_deep])
css = condense_spectra(cf.spectra[sub_idxs_shal,:], cf.nxs[sub_idxs_shal])
# normalize both of them
csd[3:3:end] ./= length(sub_idxs_deep)
css[3:3:end] ./= length(sub_idxs_shal)

begin
rosd = annulus_heatmap(csd, length(csd); r0=2.0, ep=1.0, cmax=25.0); 
savefig(rosd,"Testing/c$(case)/ans_deep.png")
ross = annulus_heatmap(css, length(css); r0=2.0, ep=1.0, cmax=25.0);
savefig(ross,"Testing/c$(case)/ans_not_deep.png")
end

# annuli by quadrant
for i = 1:nbins
    subdeep_i = subsample_safe(idx_bins_deep[i], N)
    subshal_i = subsample_safe(idx_bins_shallow[i], N)
    csd_i = condense_spectra(cf.spectra[subdeep_i,:], cf.nxs[subdeep_i])
    css_i = condense_spectra(cf.spectra[subshal_i,:], cf.nxs[subshal_i])
    csd_i[3:3:end] ./= length(subdeep_i)
    css_i[3:3:end] ./= length(subshal_i)

    degs = Int(round((i-0.5)*360/nbins))
    rosd_i = annulus_heatmap(csd_i, length(csd_i); r0=2.0, ep=1.0, cmax=25.0); 
    savefig(rosd_i,"Testing/c$(case)/ans_deep_$(degs)_low.png")
    ross_i = annulus_heatmap(css_i, length(css_i); r0=2.0, ep=1.0, cmax=25.0);
    savefig(ross_i,"Testing/c$(case)/ans_not_deep_$(degs)_low.png")
end