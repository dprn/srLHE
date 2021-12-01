module SrLHE

export sR_lhe, sR_wc, SE2_lhe, SE2_wc, Result

using Images, ImageFiltering, OffsetArrays, FFTW
using ProgressMeter

include("../../PolynomialApprox/src/PolynomialApprox.jl")
using .PolynomialApprox

include("../../SE2Conv/src/SE2Conv.jl")
using .SE2Conv

include("lift.jl")
include("sr-heat.jl")

Lift{T} = Array{T,3}
Kern{T,n} = OffsetArrays.OffsetArray{T, n, Array{T,n}}

@enum Algo LHE WC
@enum ConvType GroupConv HeatEq

struct Result{n}
    res :: Array{Float64, n}
    iter :: Int
    tol :: Float64
end
Base.getindex(R::Result, etc...)  = getindex(R.res, etc...)
iterations(R::Result) = R.iter
tolerance(R::Result) = R.tol
project(R::Result{3}; etc...) = Result{2}(project(R.res; etc...), iterations(R), tolerance(R))
Base.show(R::Result{2}) = Gray.(R[:,:])
Base.ndims(R::Result{n}) where n = n

normalize(R::Result) = Result{ndims(R)}(normalize(R.res), R.iter, R.tol)
normalize(x::Array) = typeof(x)(x - ones(size(x))*minimum(x))/(maximum(x)-minimum(x))

function project(F::Lift; normalized::Bool = true, args...)
    x = dropdims(sum(F, dims = 3), dims = 3)
    normalized ? x = normalize(x) : nothing
    return x
end

LMA(σμ::Real, I0::Array{T, 2}) where T = imfilter(I0, Kernel.gaussian(σμ) |> reflect) # convolution not correlation

function gradient_descent(I0::Array, W, λ::Real, lma::Array; 
                            Δt::Real = .15, threshold::Real = .01, max_iter::Int = 50, M::Real = 1, args...)
    # threshold : the iterations stops when two successive terms have a weighted L2 difference smaller than this quantity
    # max_iter: maximum number of iterations
    next(Ik) = (1-(1+λ)Δt)Ik + (lma + W(Ik)/(2*M))Δt + λ*Δt*I0
    proj(x) = sum(x, dims = 3) # it does nothing if x has 2 dims
    diff(a,b) = sqrt( sum(( proj(a-b) ).^2)/sum(proj(a).^2) )

    prec = I0
    cur = next(I0)
    i = 1
    
    final_iter = max_iter
    @showprogress 1 "Computing..." for i in 1:max_iter
        if diff(prec, cur) < threshold 
            final_iter = i-1
            break
        end
		# println("iteration (max. ", max_iter, "):", i)
        (prec, cur) = (cur, next(cur))
    end

    Result{ndims(I0)}(cur, final_iter, diff(prec, cur))
end

function R(I, conv, β, τ; pol_coeffs = coeffs_sigma(5, 8), args...)
    n = length(pol_coeffs)
    a = [sum([(-1.)^(j-i+1)*pol_coeffs[j+1]*binomial(j,i)*I.^(j-i) for j in i:(n-1)]) for i in 0:(n-1)]
    sum([ a[i] .* conv(I.^(i-1), β, τ; args...) for i in 1:n])
end

function lhe(I0, α, β, σw, λ, lma; sigma_interp_order = 8, conv_type::ConvType = HeatEq, args...)
    σ(x) = min(max(α*x, -1), 1) # LHE sigma function
    c = coeffs_approx(σ, sigma_interp_order)
    if conv_type == HeatEq
        res = gradient_descent(I0, I -> R(I, sR_heat, β, σw, pol_coeffs = c), λ, lma2; args...)
    elseif conv_type == GroupConv
        # we need odd number of pixels
        init = padarray(I0, Fill(0,(1,0,0),(0,1,0))) |> parent |> unfold
        lma2 = padarray(lma2, Fill(0,(1,0,0),(0,1,0))) |> parent |> unfold
        G = GridSE2(size(init, 1), size(init, 3))
        kern = sr_heat(G, t = σw/100)
        group_conv(f, β, σw; args...) = corr(f, kern)
        res = gradient_descent(init, I -> R(I, group_conv, β, σw, pol_coeffs = c), λ, lma2; args...)
    end
    res
end

function wc(I0, α, β, σw, λ, lma; conv_type::ConvType = HeatEq, args...)
    σ(x) = -min(max(α*(x-1/2), -1), 1) # WC sigma function
    if conv_type == HeatEq
        res = gradient_descent(I0, I -> sR_heat(σ.(I), β, σw; args...), λ, lma; args...)
    elseif conv_type == GroupConv
        # we need odd number of pixels
        init = padarray(I0, Fill(0,(1,0,0),(0,1,0))) |> parent |> unfold
        lma2 = padarray(lma, Fill(0,(1,0,0),(0,1,0))) |> parent |> unfold
        G = GridSE2(size(init, 1), size(init, 3))
        kern = sr_heat(G, t = σw/100)
        res = gradient_descent(init, I -> corr(σ.(I), kern), λ, lma2; args...)
    end
    res
end

sR_lhe(I0, β, σμ, σw, λ, MS; args...) = sR_evo(LHE, I0, β, σμ, σw, λ, MS;  args...)
sR_wc(I0, β, σμ, σw, λ, MS; args...) = sR_evo(WC, I0, β, σμ, σw, λ, MS; args...)

SE2_lhe(I0, β, σμ, σw, λ, MS; args...) = sR_evo(LHE, I0, β, σμ, σw, λ, MS; conv_type = GroupConv, args...)
SE2_wc(I0, β, σμ, σw, λ, MS; args...) = sR_evo(WC, I0, β, σμ, σw, λ, MS; conv_type = GroupConv, args...)
function sR_evo(algo::Algo, I0, β, σμ, σw, λ, MS; θs = 16, α = 5, sigma_interp_order = 8, verbose = false, normalized = true, args...)
    F0 = lift(I0, θs, m = size(I0, 1))
    lma = lift(LMA(σμ, I0), θs; args...)
    
    if algo == LHE
        res = project(lhe(F0, α, β, σw, λ, lma; M = MS, args...), normalized = false)
    elseif algo == WC
        res = project(wc(F0, α, β, σw, λ, lma; M = MS, args...), normalized = false)
    end

    normalized ? (res = normalize(res)) : nothing

    if verbose 
        res 
    else 
        println("Total iterations:", iterations(res), ". Final tolerance: ", tolerance(res))
        show(res)
    end
end

end # module
