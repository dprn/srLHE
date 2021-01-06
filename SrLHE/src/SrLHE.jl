module SrLHE

export sR_lhe, sR_wc, Result

using Images, ImageFiltering, OffsetArrays, FFTW
using ProgressMeter

include("../../PolynomialApprox/src/PolynomialApprox.jl")
using .PolynomialApprox

include("lift.jl")
include("sr-heat.jl")

Lift{T} = Array{T,3}
Kern{T,n} = OffsetArrays.OffsetArray{T, n, Array{T,n}}

@enum Algo LHE WC

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

function R(I, β, τ; pol_coeffs = coeffs_sigma(5, 8), args...)
    n = length(pol_coeffs)
    a = [sum([(-1.)^(j-i+1)*pol_coeffs[j+1]*binomial(j,i)*I.^(j-i) for j in i:(n-1)]) for i in 0:(n-1)]
    sum([ a[i] .* sR_heat(I.^(i-1), β, τ; args...) for i in 1:n])
end


function lhe(I0, α, β, σw, λ, lma; sigma_interp_order = 8, args...)
    σ(x) = min(max(α*x, -1), 1) # LHE sigma function
    c = coeffs_approx(σ, sigma_interp_order)
    W(I) = R(I, β, σw, pol_coeffs = c)
    gradient_descent(I0, W, λ, lma; args...)
end

function wc(I0, α, β, σw, λ, lma; args...)
    σ(x) = -min(max(α*(x-1/2), -1), 1) # WC sigma function
    W(I) = sR_heat(σ.(I), β, σw; args...)
    gradient_descent(I0, W, λ, lma; args...)
end


sR_lhe(I0, β, σμ, σw, λ, MS; args...) = sR_evo(LHE, I0, β, σμ, σw, λ, MS; args...)
sR_wc(I0, β, σμ, σw, λ, MS; args...) = sR_evo(WC, I0, β, σμ, σw, λ, MS; args...)
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
