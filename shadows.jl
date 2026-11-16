include("L81.jl")     # includes the L81 propagator

function shadows(col::ColumnProfile,waves::AbstractArray{WaveProfile})
    #=
    Computes the fraction of a profile that is shadowed according to the Lindzen '81 
    gravity wave parameterization scheme -- i.e., what fraction of the z extent of
    the column cannot see L81 waves break if they're launched from some given launchlevel

    For this metric to be sensible all waves should be launched at the same altitude
    =#
    tmp = zeros(2*col.nlev)
    shade = trues(2*col.nlev)
    for i = eachindex(waves)
        L81_wp!(waves[i],col,tmp)
        shade .= (shade .&& (tmp .== 0.0))  # tmp should be mostly exactly 0 each wave
    end
    return shade
end

function shadow_runner(vars,waves)
    #=
    Runs shadows on a dataset defined by vars

    For this metric to be sensible all waves should be launched at the same altitude
    =#
    
    # vars is a dictionary of variables
    # each variable is a vector of 4D arrays corresponding to a simulation
    # the dimensions are x,y,z,t
    nz = size(vars["Up"][1],3)
    shadow_depth = zeros(2nz)
    shadow_drag = zeros(2nz)
    total_drag = zeros(2nz)
    
    nf = length(vars["Up"])
    count = 0
    for fidx = 1:nf
        println("Beginning file $fidx")
        
        U = vars["Up"][fidx]
        V = vars["Vp"][fidx]
        ugw = vars["DX"][fidx]
        vgw = vars["DY"][fidx]
        N = vars["N"]
        rho = vars["rho"]
        sz = size(U)
        nx,ny,_,nt = sz
        for ti=1:nt, yi=1:ny, xi=1:nx
            col = ColumnProfile(U[xi,yi,:,ti], V[xi,yi,:,ti], N, rho, nz)
            momdep = cat(dims=1,ugw[xi,yi,:,ti], vgw[xi,yi,:,ti]) # concatenate x- and y-fluxes

            shade = shadows(col,waves)
            shadow_depth .+= shade                     # fraction of profiles shadowed at this height
            shadow_drag .+= shade.*(momdep.^2)         # fraction of square drag shadowed at this height
            total_drag .+= (momdep.^2)

            count += 1
        end
    end
    shadow_depth /= count
    shadow_drag .= sqrt.(shadow_drag./total_drag)
    return shadow_depth, shadow_drag
end

function latprop_shadows(U,V,rho,N,ugw,vgw,waves,radius)
    #=
    Coarse metric of the impact of lateral propagation on gravity wave drag deposition
    Computes the conventional 1D shadowing of each column, then compares the shadowing
    of near neighbors with L2 distance <= radius. If any of them is not shadowed, then
    the latprop shadowing is also not shadowed (group AND gate).

    Only practical on one file/case: piece together the results from many single files
    to get the equivalent of shadow_runner above.
    =#
    nx,ny,nz,nt = size(U)
    # We need to store all shadows per slice to avoid recomputation, 
    # but not across timeslices since we're not comparing across slices
    shadows = zeros(nx,ny,2nz)
    shadow_drag = zeros(nx,ny,2nz)

    shadow_depth = zeros(2nz)
    shadow_drag = zeros(2nz)
    for ti = 1:nt
        for yi=1:ny, xi=1:nx
            col = ColumnProfile(U[xi,yi,:,ti], V[xi,yi,:,ti], N, rho, nz)
            momdep = cat(dims=1,ugw[xi,yi,:,ti], vgw[xi,yi,:,ti]) # concatenate x- and y-fluxes
            shadows[xi,yi,:] .= shadows(col,waves)
            shadow_drag[xi,yi,:] .= shadows[xi,yi,:].*momdep
        end
    end
end