module PolynomialApprox

using Polynomials
using QuadGK

export coeffs_approx, polynomial_approx

# We consider L^2([a,b])
const a = -1
const b = 1

function canonical_basis(n)
    x = [i==j ? 1. : 0. for i in 1:n, j in 1:n]
    mapslices(Polynomial, x, dims = [2]) |> vec
end

function dot(v::Polynomial,u::Polynomial)
    P = integrate(v*u)
    P(b)-P(a)
end

function proj(v::Polynomial,u::Polynomial)
    (dot(u,v)/dot(u,u))*u
end

function gram_schmidt(b::Array{Polynomial{T, :x}}) where T<:Number
    ob = similar(b)
    for i in 1:length(b)
        v = b[i]
        for j in 1:(i-1)
            v -= proj(b[i], ob[j])
        end
        ob[i] = v/sqrt(dot(v,v))
    end
    ob
end

orthonormal_basis(n) = canonical_basis(n) |> gram_schmidt

dot(f, g) = first( quadgk( x-> f(x)g(x), a, b) )
dot(f, p::Polynomial) = dot(f, x->p(x))

function approx(f, ob::Array{Polynomial{T, :x}}) where T<:Number
    cmp = [ dot(f, ob[i]) for i in 1:length(ob) ]
    sum([cmp[i]*(ob[i]) for i in 1:length(ob)])
end

polynomial_approx(f, n::Int) = approx(f, orthonormal_basis(n))

coeffs_approx(f, n::Int) = coeffs(polynomial_approx(f, n))

end # module
