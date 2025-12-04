"""
Utility functions for Hopfield network operation: decoding, hamming distance,
energy calculation, and asynchronous recover (recall) routine.
"""

export decode, hamming, recover

function hamming(a::AbstractVector, b::AbstractVector)
    @assert length(a) == length(b)
    return sum(a .!= b)
end

function decode(s::AbstractVector{<:Integer})
	# Convert a +-1 vector into a number_of_rows x number_of_cols matrix
	N = length(s)
	@assert N == number_of_rows * number_of_cols "decode: mismatch with global image dims"
	img = Array{Float32,2}(undef, number_of_rows, number_of_cols)
	idx = 1
	for r in 1:number_of_rows
		for c in 1:number_of_cols
			img[r,c] = s[idx] == 1 ? 1.0f0 : 0.0f0
			idx += 1
		end
	end
	return img
end

function energy_of_state(model::MyClassicalHopfieldNetworkModel, s::AbstractVector)
	s_f = Float32.(s)
	return -0.5f0 * dot(s_f, model.W * s_f) - dot(model.b, s_f)
end

function recover(model::MyClassicalHopfieldNetworkModel, s0::Array{Int32,1}, true_image_energy::Float32; 
		maxiterations::Int=1000, patience::Union{Int,Nothing}=5, miniterations_before_convergence::Union{Int,Nothing}=nothing)

	N = length(s0)
	if miniterations_before_convergence === nothing
		miniterations_before_convergence = (patience === nothing) ? 1 : patience
	end

	frames = Dict{Int, Array{Int32,1}}()
	energydictionary = Dict{Int, Float32}()

	s = copy(s0)
	t = 1
	frames[t] = copy(s)
	energydictionary[t] = energy_of_state(model, s)

	history = Vector{Array{Int32,1}}()
	push!(history, copy(s))

	while t < maxiterations
		i = rand(1:N)
		# local field
		h = zero(Float32)
		for j in 1:N
			h += model.W[i,j] * Float32(s[j])
		end
		h -= model.b[i]
		s_i_new = h >= 0f0 ? Int32(1) : Int32(-1)
		s[i] = s_i_new

		t += 1
		frames[t] = copy(s)
		energydictionary[t] = energy_of_state(model, s)

		push!(history, copy(s))
		if length(history) > (patience === nothing ? 1 : patience)
			deleteat!(history, 1)
		end

		if (patience !== nothing) && length(history) == patience && t >= miniterations_before_convergence
			all_equal = all(x -> x == history[1], history)
			if all_equal
				break
			end
		end
	end

	return frames, energydictionary
end

