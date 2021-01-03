import Pkg
Pkg.activate("./SrLHE")

using Images

using SrLHE

#####################
# Global parameters #
#####################

N = 200
K = 16 # orientations
β = K/sqrt(2*N^2); # needed for the unit coherency in spatial and orientation dimensions
M = 1.

sR_lhe(img, λ, σμ, α, Δt, Δτ, τ) = SrLHE.sR_lhe(img, β, σμ, τ, λ, M, θs = K, α = α, Δt = Δt, Δτ = Δτ)
sR_wc(img, λ, σμ, α, Δt, Δτ, τ) = SrLHE.sR_wc(img, β, σμ, τ, λ, M, θs = K, α = α, Δt = Δt, Δτ = Δτ)

##################
# Gratings tests #
##################

mkpath("results")

img = Float64.(load("test_images/gratings.png"))

save("gratings_wc.png", sR_wc(img, 0.01, 6.5, 20, 0.1, 0.01, 5.))
save("gratings_lhe.png", sR_lhe(img, 2., 1., 8, 0.15, 0.01, 5.))

# Dependence w.r.t. τ

save("results/gratings_τ=0.1_lhe.png", sR_lhe(img, 2., 1., 6, 0.15, 0.01, 0.1))
save("results/gratings_τ=0.5_lhe.png", sR_lhe(img, 2., 1., 6, 0.15, 0.01, 0.5))
save("results/gratings_τ=2.5_lhe.png", sR_lhe(img, 2., 1., 6, 0.15, 0.01, 2.5))

##############################
# Original Poggendorff tests #
##############################

img = Float64.(load("test_images/original_poggendorff.png"))

normalize_target1(res) = res[75:124,85:114] |> SrLHE.normalize |> x->Gray.(x)

res = sR_lhe(img, 0.5, 2.5, 8, 0.15, 0.1, 2.5, max_iter = 10)
save("results/original_lhe.png", res)
save("results/original_lhe_zoom.png", normalize_target1(res))
