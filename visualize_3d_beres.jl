using Plots
using LinearAlgebra

function Dfun(c, U, om)
    c2 = dot(c,c)
    return om*(dot(c,U)/c2 - 1)
end

function mfun(c, U, om, N2; Dzz=0.0)
    D = Dfun(c, U, om)
    c2 = dot(c,c)
    K2 = om*om/c2
    return sqrt(K2 * (N2/D^2 - 1) - Dzz/D)
end

function flux(c, U, om, N2, h, L, th0; Dzz=0.0)
    g = 9.81

    nc = norm(c)
    K = om/nc
    D = Dfun(c, U, om)
    m = mfun(c, U, om, N2; Dzz)

    horz = pi^2*L*exp(-2L^2*pi*K^2)
    vert = (-pi*h/(m^2*h^2 - pi^2))^2
    forc = (g/th0)^2
    mass = om^2/nc^4

    return mass * forc * horz * vert * K^3 * D^(-4) * m^(-1)
end

U = [10.0 0.0]

cx = -100.0:100.0
cy = -100.0:100.0
nx = length(cx)
ny = length(cy)

L = 30e3  # 10km heating width (?)
h = 1e3   # 1km heating depth
N2 = (1e-2)^2

th0 = 300 # 300K potential temp ?

om = 2pi/(3600*2) # 2hr frequency?

Darr = zeros(nx,ny)
c = zeros(2)
for xi = 1:nx
    for yi = 1:ny
        c[1] = cx[xi]
        c[2] = cy[yi]
        Darr[xi,yi] = Dfun(c,U,om)
    end
end

marr = zeros(nx,ny)
for xi = 1:nx
    for yi = 1:ny
        c[1] = cx[xi]
        c[2] = cy[yi]
        marr[xi,yi] = 1/mfun(c, U, om, N2)
    end
end

fluxarr = zeros(nx,ny)
for xi = 1:nx
    for yi = 1:ny
        c[1] = cx[xi]
        c[2] = cy[yi]
        fluxarr[xi,yi] = flux(c, U, om, N2, h, L, th0)
    end
end

plot(cx,cy,Darr',st=:surface,xlabel="cx (m/s)",ylabel="cy (m/s)",label="D (s^-1)")
savefig("Dsurf.png")
heatmap(cx,cy,Darr',xlabel="cx (m/s)",ylabel="cy (m/s)",label="D (s^-1)")
savefig("Dmap.png")

plot(cx,cy,marr',st=:surface,xlabel="cx (m/s)",ylabel="cy (m/s)",label="1/m (m)")
savefig("msurf.png")
heatmap(cx,cy,marr',xlabel="cx (m/s)",ylabel="cy (m/s)",label="1/m (m)")
savefig("mmap.png")

plot(cx,cy,fluxarr',st=:surface,xlabel="cx (m/s)",ylabel="cy (m/s)",label="flux (Pa)")
savefig("fluxsurf.png")
heatmap(cx,cy,fluxarr',xlabel="cx (m/s)",ylabel="cy (m/s)",label="flux (Pa)")
savefig("fluxmap.png")