@everywhere function myrange(d1)
	if nprocs() == 1
		return 1:d1
	end
    idx = myid()
    if idx == 1
        # This worker is not assigned a piece
        return 1:0
    end
    nchunks = nprocs() - 1
    splits = [round(Int, s) for s in linspace(0, d1, nchunks + 1)]
    return splits[idx - 1] + 1:splits[idx]
end




@everywhere function myrange(d1)
    idx = myid()
    if idx == 1
        # This worker is not assigned a piece
        return 1:0
    end
    nchunks = nprocs() - 1
    splits = [round(Int, s) for s in linspace(0, d1, nchunks + 1)]
    return splits[idx - 1] + 1:splits[idx]
end