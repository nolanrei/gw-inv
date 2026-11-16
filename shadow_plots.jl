include("L81-2.jl")

# Makes a visualization of the regions of a column that cannot receive drag from 
# waves of a given phase velocity and flux

function bfn(c,absk,v,rho,N)
    # Lindzen breaking criterion for a wave of phase spd c
    # and wavespeed |k| in a flow v = \vec{u} dot \hat{c}
    # This is therefore the needed flux to trigger breaking here
    # (as long as this wave hasn't broken lower)
    return rho * 0.5*absk*(c-v)^3/N
end

function shadowing(col, dir; wspd_range=-60.0:60.0)
    nw = length(wspd_range)
    nz = length(col.U)
    
    sh = zeros(nw,nz)

    for j = 1:nw
        bprev = bfn(wspd_range[j],2pi/5e5,col.U[1]*dir[1] + col.V[1]*dir[2],col.rho[1],col.N[1])
        for i = 1:nz
            b = bfn(wspd_range[j],2pi/5e5,col.U[i]*dir[1] + col.V[i]*dir[2],col.rho[i],col.N[i])
            if b*bprev <= 0.0 
                # signs flipped; waves would have broken at this critical layer
                break
            end
            bprev = b
            
            sh[j,i] = abs(b)
        end
    end
    return sh
end

function make_shadow_plot(sh,wavspds,z,col,dir)
    heatmap(wavspds,z,(sh'),xlabel="wavespeed (m/s)",ylabel="z (km)",clims=(0,10e-3)) # 10 mPa, not 10 Pa
    nz = length(z)
    windspeed = zeros(nz)
    for i = 1:nz
        windspeed[i] = col.U[i]*dir[1] + col.V[i]*dir[2]
    end
    plot!(windspeed,z,label="windspeed")
    savefig("Figures/WACCM/Reconstructions/shadow_plot.png")
end