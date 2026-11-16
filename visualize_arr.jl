using Plots

# Run GR backend in headless mode so it doesn't crash on the supercomputer
ENV["GKSwstype"] = "100" 

# Use the @animate macro to capture each iteration of the loop as a frame
anim = @animate for i in 1:25
    
    # Replace the hardcoded [1, :] with [i, :]
    aflux    = 1e3 * reshape(fluxes[i, :], (nz, 2))
    afit     = 1e3 * reshape(fits[i, :], (nz, 2))
    
    # Generate the base plot
    # (Optional: I added 'Frame $i' to the title so you can see it change)
    bot_row  = plot(aflux, z, layout=(1, 2), title="$(nhrs)hr recon (Frame $i)", label="flux", 
        xlabel=["X flux (mPa)" "Y flux (mPa)"], ylabel="z (km)")
    
    # Overlay the recon data
    plot!(bot_row, afit, z, layout=(1, 2), label="recon")
    
end

# Save to file. 4 frames per second (fps) gives you 0.25 seconds between frames.
# You can also use mp4(anim, "animation.mp4", fps=4) if you prefer a video file.
gif(anim, "flux_animation.gif", fps = 4)