include("spectrum_metric.jl")

### Standard candle 0 -- spike vs itself (0 in balanced, regularization makes it nonzero)
d1 = [0.0, 10.0, 1e-3]
d2 = [0.0, 10.0, 1e-3]
sc0 = earth_mover_dist_tv(d1,d2)

### Standard candle 1 -- two spikes of 1mPa flux, offset by d m/s
d1 = [0.0, 0.0, 1e-3]
d2 = [0.0, 0.0, 1e-3]
d = [1, 5, 10, 20, 40, 80, 150]
ld = length(d)
sc1 = zeros(ld)
for i = 1:ld
    sc1[i] = earth_mover_dist_tv(d1,d2.+[0.0,d[i],0.0])
end

### Standard candle 2 -- same wave, different masses
d1 = [0.0, 0.0, 1e-3]
m = [0.2, 0.5, 0.8, 1.2, 2, 5]
lm = length(m)
sc2 = zeros(lm)
for i = 1:lm
    sc2[i] = earth_mover_dist_tv(d1, d1.*[1.0,1.0,m[i]])
end

### Standard candle 3 -- Add an extra wave
d1 = [0.0, 0.0, 1e-3]
d2 = [0.0, 0.0, 1e-3, 0.0, 10.0, 1e-3]
sc3 = earth_mover_dist_tv(d1,d2)