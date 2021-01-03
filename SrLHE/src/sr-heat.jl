sR_heat(I, β, τ; Δτ=.1, args...) = sr_heat_fourier(I, Δτ/τ, beta = τ, alpha = τ*β^2)

function sr_heat_fourier(I::Array{T,3}, Δt::Real; alpha = .1, beta = .02) where T<:Real
# Evolution over the time interval [0,1], of the equation 
# $$
# \partial_t f = (\beta X_1 + \alpha X_2^2) f
# $$
    flift = fft(I, [1,2])
    s = size(flift)

    for i = 1:size(flift,1), j = 1:size(flift,2)
            # column = squeeze(flift[i,j,:],(1,2))
            column = flift[i,j,:]
            flift[i,j,:] = cn_oneline_sd(column, i,j, s[1], s[2], alpha, beta, Δt)
	end

    ifft(flift,[1,2]) |> real
end

function cn_oneline_sd(fin::Array{T,1}, i::Int64, j::Int64, n1::Int64, n2::Int64, aalpha::Float64, bbeta::Float64, tau::Float64) where T<:Number

	sinx=sin(2*pi*(i-1)/n1)
	siny=sin(2*pi*(j-1)/n2)

	for i=0:tau:1
		fin = cn_onestep(fin,sinx,siny,n1,aalpha,bbeta,tau)
	end

	return fin
end

function cn_onestep(fin::Array{T,1},sinx::Float64,siny::Float64,n1::Int64, aalpha::Float64, bbeta::Float64, tau::Float64) where T<:Number

	n3 = length(fin)
	a = similar(fin, Float64)
	frhs = similar(fin)

	dtet=pi/n3
	for k=1:n3
		at = -bbeta*n1*(sinx*cos((k-1)*dtet)+siny*sin((k-1)*dtet))^2
		a[k] = 1 - at*tau*0.5 + aalpha*tau/(dtet^2)
	end
	b = -0.5*tau*aalpha/(dtet^2)

	for k=2:n3-1
		at=-bbeta*n1*(sinx*cos((k-1)*dtet)+siny*sin((k-1)*dtet))^2
		frhs[k]=fin[k] + 0.5*tau*(aalpha*(fin[k+1]-2*fin[k]+fin[k-1])/(dtet^2) + at*fin[k])
	end
	at=-bbeta*n1*(sinx)^2

	frhs[1]=fin[1]+0.5*tau*(aalpha*(fin[2]-2*fin[1]+fin[n3])/(dtet^2) + at*fin[1])
	at=-bbeta*n1*(sinx*cos((n3-1)*dtet)+siny*sin((n3-1)*dtet))^2
	frhs[n3]=fin[n3]+0.5*tau*(aalpha*(fin[1]-2*fin[n3]+fin[n3-1])/(dtet^2) + at*fin[n3])

	ff = linsol_per(a,b,real(frhs))
    a = linsol_per(a,b,imag(frhs))

	ff + im*a
end

function linsol_per(c::Array{Float64,1}, a::Float64,f::Array{Float64,1})
	#  solve linear systems of the form
   #
	#  c a 0 0 ... 0 0 a  x(1)   f(1)
	#  a c a 0 ... 0 0 0  x(2)   f(2)
	#  0 a c a ... 0 0 0  x(3) = f(3)
	#          ...        ...    ...
	#  a 0 0 0 ... 0 a c  x(n)   f(n)

	@assert( length(c) == length(f) )

	n = length(c)

	alp = zeros(Float64, n+1)
	bet = zeros(Float64, n+1)
	gam = zeros(Float64, n+1)

	p = zeros(Float64, n-1)
	q = zeros(Float64, n-1)

	b=a

	alp[2]=-b/c[1]
	bet[2]=f[1]/c[1]
	gam[2]=-a/c[1]

	for i=2:n
		denom =(c[i]+alp[i]*a)
		alp[i+1] = -b/denom
		bet[i+1] = -(-f[i]+a*bet[i])/denom
		gam[i+1] = -a*gam[i]/denom
	end

	p[n-1] = bet[n]
	q[n-1] = alp[n] + gam[n]

	for i=n-2:-1:1
		p[i]=alp[i+1]*p[i+1]+bet[i+1]
		q[i]=alp[i+1]*q[i+1]+gam[i+1]
	end


	x = zeros(Float64, n)
	x[n]=(p[1]*alp[n+1]+bet[n+1])/(1-gam[n+1]-q[1]*alp[n+1])
	for i=1:n-1
		x[i]=p[i]+x[n]*q[i]
	end

	return x
end
