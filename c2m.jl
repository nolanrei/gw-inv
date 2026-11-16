using Polynomials
include("L81-2.jl")

function c_to_wave(cx::Float64, cy::Float64, dir::AbstractArray{Float64}, K::Float64, usrc::AbstractArray{Float64}, N2::Float64)
    # Neglecting f and 1/4H^2 in the dispersion relation
    # dir is the direction of (k,l) -- should be equal to direction of c,
    #      but can be nonzero while c is 0 (lee waves)
    # K^2 = k^2 + l^2 magnitude of wavenumber vector
    # cx,cy are phase speeds
    # usrc is the meanflow velocity at the wave source level
    # N2 is squared buoyancy frequency at source level
    K2 = K*K
    N = sqrt(N2)
    C2 = cx*cx + cy*cy
    C = sqrt(C2)

    @assert (abs(abs(cx*dir[1]+cy*dir[2]) - C) < 1e-14) "dir should be parallel to (cx, cy)"
    
    k = K*sign(cx*dir[1])*dir[1]
    l = K*sign(cy*dir[2])*dir[2]

    UK = usrc[1]*k + usrc[2]*l
    m = N*K/abs(K*C - UK)    # L81 WKB m evaluated at source level
    kl2 = K2+m*m
    
    omint = N*K/sqrt(kl2)    # intrinsic freq
    omabs = omint + UK       # absolute frequency
    cz = omabs*m/kl2         # phase speed z
    println(cx,"\t",omabs*k/kl2)
    wav = WaveProfile(k,l,m,omabs,MVector{3}([cx,cy,cz]),0.0,0)   # set the flux and source level later
    return wav
end

# test case
dir = [1.0, -1.0]
dir ./= norm(dir)
usrc = [10.0, 5.0]
cx = 10.0
cy = -10.0
K = 2pi/50000    # wavelength 50km
N2 = (2e-2)^2

wav = c_to_wave(cx,cy,dir,K,usrc,N2)