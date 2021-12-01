module SE2Conv

using OffsetArrays, ImageFiltering

export Point, Angle, SE2, w_bek, GridSE2, GroupConvKernel, OrientedKernel, corr, sr_heat

##########################
# Group structure of SE2 #
##########################

# Type for points of $\mathbb R^2$

struct Point{T <: Real}
    x::T
    y::T
end

Point(x::Real, y::Real) = Point(promote(x,y)...)
Base.show(io::IO, p::Point) = print(io, "(", p.x ," , ", p.y, ")")

Base.:+(p::Point, q::Point) = Point(p.x + q.x, p.y + q.y)
Base.:-(p::Point, q::Point) = Point(p.x - q.x, p.y - q.y)
Base.:-(p::Point) = Point(-p.x, -p.y)

Base.zero(::Type{Point{T}}) where T<:Real = Point{T}(0,0)
Base.zero(::Type{Point}) = zero(Point{Float64})

# Type for angles in [-π, π]

# mod in [-π,π]
modpi(x) = rem2pi(x,RoundNearest)

struct Angle{T<:Real} <: Number
    θ::T
    Angle(x::T) where {T} = new{T}(modpi(x))
    Angle{T}(x::S) where {T <:Number, S<:Number} = new(convert(T,modpi(x)))
end

Base.show(io::IO, θ::Angle) = print(io, θ.θ)

Base.:+(θ::Angle, ϕ::Angle) = Angle(θ.θ + ϕ.θ)
Base.:-(θ::Angle, ϕ::Angle) = Angle(θ.θ - ϕ.θ)
Base.:-(θ::Angle) = Angle(-θ.θ)

Base.:/(θ::Angle, x::Number) = Angle(θ.θ/x)
Base.:*(θ::Angle, x::Number) = Angle(θ.θ * x)

Base.cos(θ::Angle) = cos( θ.θ )
Base.sin(θ::Angle) = sin( θ.θ )
Base.tan(θ::Angle) = tan( θ.θ )

Base.:*(θ::Angle, p::Point) = Point( cos(θ)*p.x - sin(θ)*p.y, sin(θ)*p.x + cos(θ)*p.y )

Base.zero(::Angle{T}) where T<:Real = Angle(zero(T))
Base.real(T::Angle{R}) where R<:Real = R
Base.promote_rule(::Type{Angle{T}}, ::Type{S}) where {T<:Real, S<:Real} = Angle{promote_type(T, S)}

Base.isapprox(x::Angle, y::Number; args...) = isapprox(x.θ, y; args...)
Base.isapprox(x::Angle, y::Angle; args...) = isapprox(x.θ, y.θ; args...)

# Type for SE2 points

struct SE2{T1<:Real, T2<:Real}
    p::Point{T1}
    θ::Angle{T2}
end

pt(a::SE2) = a.p
x(a::SE2) = a.p.x
y(a::SE2) = a.p.y
θ(a::SE2) = a.θ.θ

SE2(p::Point, θ::Real) = SE2(p, Angle(θ))
SE2(x::Real, y::Real, θ::Real) = SE2(Point(x,y), Angle(θ))
Base.show(io::IO, a::SE2) = print(io, "(", x(a) ," , ", y(a), " , " , θ(a), ")")

Base.:+(a::SE2, b::SE2) = SE2( a.p + a.θ*b.p, θ(a) + θ(b) )
Base.:-(a::SE2) = SE2( -((-a.θ)*a.p), -θ(a) )
Base.:-(a::SE2, b::SE2) = a+(-b)

Base.isapprox(a::SE2, b::SE2) = isapprox(x(a), x(b), atol = eps()) * isapprox(y(a), y(b), atol = eps()) * (θ(a) ≈ θ(b))

Base.zero(::Type{SE2{T,S}}) where {T<:Real, S<:Real} = SE2(zero(Point{T}), zero(Angle{S}))
Base.zero(::Type{SE2}) = zero(SE2{Float64,Float64})

###############################
# Approximation of the kernel #
###############################

# We use the nilpotent approximation also used in
# Bekkers, Chen, and Portegies, ‘Nilpotent Approximations of Sub-Riemannian Distances
# for Fast Perceptual Grouping of Blood Vessels in 2D and 3D’.

norm_c(c1,c2,c3; ζ=16) = (( c1^2 + c2^2 )^2 + ζ*c3^2)^(1/4);

k_bek(x, y, θ; args...) = k_bek(SE2(x,y,θ); args...)
k_bek(p::Point, θ; args...) = k_bek(SE2(p,θ); args...)
"""
    k_bek(a::SE2; ζ::Real = 16) -> k_bek

Nilpotent approximation of the distance from the origin of the point a. The ζ
parameter is the weight applied to the non-horizontal direction.
"""
function k_bek(a::SE2; ζ=16)
    c1 = θ(a)
    c2 = c1 !=0 ? 0.5*c1*(y(a)+x(a)*cot(c1/2)) : x(a)
    c3 = c1 !=0 ? 0.5*c1*(-x(a)+y(a)*cot(c1/2)) : y(a)
    norm_c(c1,c2,c3; ζ = ζ)
end

"""
    w_bek(a::SE2; t = .1, ...) -> w_bek

The approximation of the heat kernel at time `t` computed according to Varadhan
formula, via the nilpotent approximation `k_bek`.
"""
w_bek(a::SE2; t = .1, args...) = exp(-k_bek(a; args...)^2/(4t))
w_bek(x, y, θ; args...) = w_bek(SE2(x,y,θ); args...)

"""
    cut_bek(t, ε; ζ = 16) -> M, ν

The approximation of the heat kernel given by `w_bek` at time `t` is guaranteed
to be smaller than the tolerance `ε` for all points outside the box

    S = [-M, M] × [-M, M] × [-ν, ν]

"""
function cut_bek(t,ε; ζ = 16)
    @assert t > 0 && 0<ε<1
    ν = 2*sqrt(-t*log(ε))
    M = max( ν, -log(ε)*4t/sqrt(ζ) )*sqrt(2)

    M, ν
end

################
# Grids in SE2 #
################

struct GridSE2{T1<:AbstractFloat, T2<:AbstractFloat} <: AbstractArray{SE2{T1,T2},3}
    x :: AbstractRange{T1}
    y :: AbstractRange{T1}
    θ :: AbstractRange{T2}
end

Base.size(G::GridSE2) = (length(G.x), length(G.y), length(G.θ))
Base.getindex(G::GridSE2, i1::Int, i2::Int, i3::Int) = SE2(G.x[i1],G.y[i2],G.θ[i3])
setindex!(G::GridSE2, v, I::Vararg{Int, 3}) = nothing

min_k(G::GridSE2) = minimum( map(k_bek, G))

"""
    GridSE2(length_x::Int, length_θ::Int; max_x=1) -> G

Creates an SE2 grid G containing the origin, with square spatial grid.
The dimensions of G are `(l, l, k)` where `l` and `k` are the nearest odd
integers bigger than `length_x` and `length_θ`, respectively.
"""
function GridSE2(length_x::Int, length_θ::Int; max_x=1)
    xs = balanced_interval(max_x, length_x)
    θs = balanced_interval(Float64(π), length_θ,)
    GridSE2(xs,xs,θs)
end


"""
    balanced_interval(max, l::Int = 0; step::Real = 0) -> I

Returns a discretization I of [-max, max] symmetric w.r.t., and containing the, 0,
such that `length(I)` equals `l` or `l+1` (if `l` is even). If the parameter `step` is
specified, it ignores `l` and computes automatically the length of the interval.
"""
function balanced_interval(max, l::Int = 0; step::Real = 0)
    @assert l != 0 || step != 0
    if step == 0
        w = isodd(l) ? l>>1 : (l+1) >> 1
        return -max*(w*2/l):2*max/l:(w*2/l)*max
    else
        w = floor(Int, max/step)
        return -w*step : step : w*step
    end
end

############################
# Group convolution in SE2 #
############################

# To perform efficiently the group convolutions, we use the algorithm proposed in
#
#     Cohen and Welling, ‘Group Equivariant Convolutional Networks’
#
# Namely, we exploit the fact that the _correlation_ of `f` with `w` in SE2 can be
# re-written as
# $$
# f\star w(\cdot, \theta) = \sum_{\varphi\in SE(2)} f_\varphi\star_{\mathbb R^2} [R_{-\theta}w_\varphi].
# $$
# Here, we let $f_\varphi(x) = f(x,\varphi)$, and $R_{-\theta}$ being the
# counter-clockwise rotation of $-\theta$.
# Hence, we need only to stock the family of 2D filters
# $$
# W : \theta \mapsto \{ R_{-\theta}w_\varphi\}_{\varphi\in I}.
# $$

"""
The filter for a fixed θ.

In the dictionary the value at the integer j yields the 2D
filter given by

    R_{-θ} w_{G.θ[j]}(⋅),

for a given SE2 grid G.
"""
struct OrientedKernel{T <: AbstractFloat}
    W :: Dict{Int, OffsetMatrix{T, Matrix{T}}}
end

"""
    corr(f::Array{<:Real, 3}, w::OrientedKernel) -> g

Being `w` an oriented filter, we return
```math
    g(x) = \\sum_{\\varphi} f(\\cdot, \\varphi)\\star_{\\mathbb R^2} [R_{-\\theta}w_\\varphi].
```
"""
@inline corr(f::Array{<:Real, 3}, w::OrientedKernel) =
    sum([ imfilter( f[:,:,k], w.W[k] ) for k in keys(w.W) ])

"""
The full filter. This is saved as a vector W such that
W[k] is the filter corresponding to the orientation θ = G.θ[k].
Here, G is an SE2 grid, that we save inside the type.
"""
struct GroupConvKernel{T <: AbstractFloat}
    Ws :: Vector{OrientedKernel{T}}
    G :: GridSE2
end

"""
    corr(f::Array{<:Real,3}, W::GroupConvKernel) -> g

Correlation between the SE2 function `f` and the convolution kernel saved in
`W`.
"""
function corr(f::Array{<:Real, 3}, W::GroupConvKernel)
    @assert size(W.G) == size(f)
    g = similar(f)
    @simd for k in 1:size(f, 3)
        g[:,:,k] = corr(f, W.Ws[k])
    end
    g
end

# Constructors

"""
Returns the index of the central element, if the array has odd length
"""
function mid(a::AbstractArray)
    @assert isodd(length(a))
    convert(Int, (length(a)+1)/2)
end

"""
    OrientedKernel(G::GridSE2, w, cutoff::Real, k::Int = mid(G.θ)) -> W_θ

Constructs an `OrientedKernel` on the grid `G`, corresponding to the angle `θ =
G.θ[k]`. The function used to construct the kernel is the argument `w`, which is
assumed to be significative only inside the bounds given by `cutoffs`.
"""
@inline function OrientedKernel(G::GridSE2, w, cutoffs::Tuple{Real, Real}, k::Int = mid(G.θ))
    xs = balanced_interval(cutoffs[1], step = step(G.x))
    ys = balanced_interval(cutoffs[1], step = step(G.y))

    θ = G.θ[k]

    # We can only compute the filter along these angles (as given by the `cutoff` parameter
    ks = mod1.( (mid(G.θ)-ceil(Int, cutoffs[2]/step(G.θ))):(mid(G.θ)+ceil(Int, cutoffs[2]/step(G.θ))) , length(G.θ) )

    g = [ w( SE2(Angle(-θ)*Point(x,y), G.θ[j]) ) for x in xs, y in ys, j in ks ]
    # g /= sum(g)

    # When we stock the dictionary we need to remember to account for the translation
    # of \theta, in the keys.
    OrientedKernel(Dict(mod1(ks[j]+(k-mid(G.θ)), length(G.θ)) => centered(g[:,:,j]) for j in 1:length(ks)))
end
OrientedKernel(G::GridSE2, w, cutoff::Real, k::Int = mid(G.θ)) = OrientedKernel(G, w, (cutoff, cutoff) , k)

"""
Constructs the `GroupConvKernel` given by a function `w` which can only be considered
in an box of sides `cutoffs`.
"""
GroupConvKernel(G::GridSE2, w, cutoffs::Tuple{Real, Real}) = GroupConvKernel([ OrientedKernel(G, w, cutoffs, j) for j in 1:length(G.θ) ], G)
GroupConvKernel(G::GridSE2, w, cutoff::Real) = GroupConvKernel(G, w, (cutoff, cutoff))

##############################
# Sub-Riemannian heat kernel #
##############################

"""
Constructs the sR heat kernel, to be used as a group convolution kernel. Observe
that we pass the function x↦w(-x) so that the correlation with this kernel
corresponds to the convolution with w.
"""
sr_heat(G::GridSE2; ε = 1e-3, t = .05) = GroupConvKernel(G, x -> w_bek(-x; t = t), cut_bek(t, ε))

end # module
