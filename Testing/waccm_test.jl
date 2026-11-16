include("../omp.jl")
include("../spectrum_recovery.jl")
include("../ensemble_accuracy.jl")

# You need to load in a dataset before
ds = waccm_ds
z = mean_existing_dims(ds,:z,:z)
nz = length(z)

function retrieve_waccm_flux_type!(fluxvec::Vector{Float64}, ds::DimStack, 
        xi::Int, yi::Int, ti::Int,
        os::OMPStruct, flux_type="OFC")
    nz = div(length(fluxvec),2)
    _o = contains(flux_type, "O")
    _f = contains(flux_type, "F")
    _c = contains(flux_type, "C")
    if _o && _f && _c
        # then it contains "OFC" and it's easier to use the pre-summed fluxes
        for zi = 1:nz
            fluxvec[zi]    = ds[:UW][xi,yi,zi,ti]
            fluxvec[zi+nz] = ds[:VW][xi,yi,zi,ti]
        end
    else
        for zi = 1:nz
            fluxvec[zi]    = 0.0
            fluxvec[zi+nz] = 0.0
            if _o
                fluxvec[zi]    += ds[:UW_O][xi,yi,zi,ti]
                fluxvec[zi+nz] += ds[:VW_O][xi,yi,zi,ti]
            end
            if _f
                fluxvec[zi]    += ds[:UW_F][xi,yi,zi,ti]
                fluxvec[zi+nz] += ds[:VW_F][xi,yi,zi,ti]
            end
            if _c
                fluxvec[zi]    += ds[:UW_C][xi,yi,zi,ti]
                fluxvec[zi+nz] += ds[:VW_C][xi,yi,zi,ti]
            end
        end
    end
    return nothing
end

src = 1

nsamp = 400

cases = 0
case_mats = []

## Case 1 -- orographic fluxes
retrieve_func! = retrieve_waccm_flux_type!
retrieve_args = ("O",)
append!(case_mats, (retrieve_func!, retrieve_args))
cases += 1

## Case 1 -- frontal fluxes
retrieve_func! = retrieve_waccm_flux_type!
retrieve_args = ("F",)
append!(case_mats, (retrieve_func!, retrieve_args))
cases += 1

## Case 1 -- convective fluxes
retrieve_func! = retrieve_waccm_flux_type!
retrieve_args = ("C",)
append!(case_mats, (retrieve_func!, retrieve_args))
cases += 1

## Case 1 -- all fluxes
retrieve_func! = retrieve_waccm_flux_type!
retrieve_args = ("OFC",)
append!(case_mats, (retrieve_func!, retrieve_args))
cases += 1

bic_opts = BICOptions(0.0,1.0,0.0)

for case in 1:cases
    results = ensemble_accuracy(ds, retrieve_func!; 
            src, reg=1e-8, nsamp, kw=2pi/1e4, bic_opts, retrieve_args, project=true)
    save_ensemble_to_netcdf("ensemble_results_waccm$(case).nc", results)
end