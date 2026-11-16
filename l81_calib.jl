# intended to be run after something setting up columns

function L81_wp(wav::WaveProfile, col::ColumnProfile, momdep::AbstractArray{<:Number})
    # Compute gravity wave flux using the Lindzen '81 propagation scheme
    # "Lindzen '81 WavePacket formulation" to differentiate from wavespeed formulation below
    # Output is a vector of dimensions (2*nlev, 1) with [Dx; Dy]
    # Nonallocating version -- ADDS ON TO provided momdep vector

    flux = wav.flux
    absk = abs(wav.k)
    c = norm(wav.c[1:2])                     # wavespeed in the horizontal
    cdir = (c > 0 ? wav.c/c : [1 0 0])       # direction of wave propagation
    vb = col.U[wav.src]*cdir[1] + col.V[wav.src]*cdir[2]        # mean flow speed in direction of wave at src
    bb = 0.5*absk*(c-vb)^3/col.N[wav.src]

    b = zeros(col.nlev)
    b[wav.src] = bb
    
    # Loop from bottom to top of column, depositing momentum where it exceeds maximum stable transport
    for lvl = wav.src:col.nlev-1
        if flux == 0
            # if the wave has no momentum we don't need to keep going
            break
        end
        vt = col.U[lvl+1]*cdir[1] + col.V[lvl+1]*cdir[2]                   # flow speed in direction cdir at top of cell

        bt = col.rho[lvl+1]/col.rho[wav.src] * 0.5*absk*(c-vt)^3/col.N[lvl+1]      # breaking condition -- maximum stable momentum transport

        # If c-v at bottom and top of cell have different signs, there's a 
        # critical layer somewhere in here and we need to break and dump all momentum
    	if bb*bt <= 0
            bb = 0
            bt = 0   # cuts off any further deposition
        end
        b[lvl+1] = bt

        # If |flux| > b, set flux to b and deposit extra at this level
        bb = abs(bb)              # we don't care about the sign
    	momsign = sign(c-vb)      # super important -- should break below crit layers and not above (vb vs vt)
        if abs(flux) > bb    # this should probably only work if flux is positive
            fb = flux-sign(flux)*bb
            # we assume that momentum gets deposited in the direction of phase speed
            momdep[lvl] += momsign*fb*cdir[1]/col.rho[lvl]                    # zonal
            momdep[lvl+col.nlev] += momsign*fb*cdir[2]/col.rho[lvl]           # meridional (all the ys at the end)
            flux = bb
        end
        vb = vt
        bb = bt     # since we're overwriting bb, we can keep using it to detect crit layers despite abs'ing it earlier
    end

    return b
end

ref_flux = 1e-3
dir = [1.0;0.0]
dir2 = [0.0;1.0]
wavspd = 20.0*randn()
k = 1e4/(2pi)
c = SVector(wavspd*dir[1], wavspd*dir[2], 10.0)
testwave = WaveProfile(2pi/3000, 0.0, 0.0, 0.0, c, ref_flux, srcs[1])

momdep_tst = zeros(2nz)
b = L81_wp(testwave,col,momdep_tst)
plot(log10.(abs.(b[1:nz])),z,label="b")
plot!(log10.(1e-9.+abs.(momdep_tst[1:nz])),z,label="momdep")
savefig("tst.png")