import Base: size

"""
build(::Type{MyClassicalHopfieldNetworkModel}, params::NamedTuple)

Construct a `MyClassicalHopfieldNetworkModel` from a named tuple containing
`memories` (an N x K matrix where each column is a +-1 memory vector).
"""
function build(::Type{MyClassicalHopfieldNetworkModel}, params::NamedTuple)
	@assert haskey(params, :memories) "Expected named tuple with key :memories"
	memories = params[:memories]
	N, K = size(memories)

	# Convert memories to Float32 for computation
	S = Array{Float32,2}(undef, N, K)
	for k in 1:K
		S[:,k] = Float32.(memories[:,k])
	end

	# Hebbian outer-product sum (average)
	W = zeros(Float32, N, N)
	for k in 1:K
		s = S[:,k]
		W .+= s * s'
	end
	W ./= Float32(K)

	# no self-connections
	for i in 1:N
		W[i,i] = 0f0
	end

	b = zeros(Float32, N)

	# compute energy for each stored memory
	energy = Vector{Float32}(undef, K)
	for k in 1:K
		s = S[:,k]
		energy[k] = -0.5f0 * dot(s, W * s) - dot(b, s)
	end

	return MyClassicalHopfieldNetworkModel(W, b, energy)
end

export build
