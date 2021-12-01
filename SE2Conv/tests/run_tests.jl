using Test

using SE2Conv

#############################################
# Test of correlation with Dirac δ function #
#############################################

G = GridSE2(64, 16)

# We need to use ≈ instead than = ! Indeed,
# SE2(-0,0,0) != SE2(0,0,0).
δ(a::SE2) = a ≈ zero(SE2) ? 1. : 0.

@test begin
    f = rand(size(G)...)
    f == corr(f, GroupConvKernel(G, δ, 1e-1))
end


################################################################################
# Convolution of sR heat kernel with δ, and test to see if it is indee `w_bek` #
################################################################################

f = [ δ(a) for a in G ]

for τ in 1e-3:1e-2:1e-1
    print("Testing τ = ", τ, " -> ")
    W = sr_heat(G, t = τ);
    ww = [ w_bek(x,y,θ, t = τ) for x in G.x, y in G.y, θ in G.θ ]
    println(maximum(abs.(corr(f, W) - ww)) < 1e-3 )
end

