mutable struct MyClassicalHopfieldNetworkModel
	W::Array{Float32,2}      # weight matrix (N x N)
	b::Array{Float32,1}      # bias vector (N,)
	energy::Vector{Float32}  # energy value for each stored memory
end

export MyClassicalHopfieldNetworkModel