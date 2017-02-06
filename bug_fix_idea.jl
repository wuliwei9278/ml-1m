l = convert(SharedArray, zeros(nprocs()))

res = zeros(4)
res = @parallel (+) for i = 1:10
	a = zeros(4)
	a[1] = i
	@show a
end


@everywhere function myrange(loop_numbers)
    idx = myid()
    if idx == 1
        # This worker is not assigned a piece
        return 1:0
    end
    nchunks = nprocs() - 1
    splits = [round(Int, s) for s in linspace(0,loop_numbers,nchunks+1)]
    return splits[idx - 1]+1:splits[idx]
end

@everywhere function copy!(res, a)
	for i in 1:4
		res[i] += a[i]
	end
end

@everywhere function chunk!(res, irange)
    @show (irange)  # display so we can see what's happening
    a = zeros(4)
    for i in irange
        for j in 1:4
        	a[j] += i
        end
    end
    copy!(res, a)
    #@show res
    return res
end

@everywhere shared_chunk!(res, loop_numbers) = chunk!(res, myrange(loop_numbers))

function shared!(res, loop_numbers)
    @sync begin
        for p in 1:nprocs()
            @async remotecall_wait(p, shared_chunk!, res, loop_numbers)
        end
    end
    res
end

loop_numbers = 1000000
res = SharedArray(Float64, (4))
@time shared!(res, loop_numbers)


julia> @time shared!(res, loop_numbers)
irange = 1:0	From worker 2:	irange = 1:250000
	From worker 3:	irange = 250001:500000

	From worker 4:	irange = 500001:750000
	From worker 5:	irange = 750001:1000000
  0.003344 seconds (2.11 k allocations: 162.266 KB)
4-element SharedArray{Float64,1}:
 5.00001e11
 5.00001e11
 5.00001e11
 5.00001e11

function parallel!(res, loop_numbers)
	res = @parallel (+) for i = 1:loop_numbers
		a = zeros(4)
		for j in 1:4
        	a[j] = i
        end
		#@show a
		a
	end
	return res
end

res = zeros(4)
@time parallel!(res, loop_numbers)

julia> @time parallel!(res, loop_numbers)
  0.066190 seconds (6.64 k allocations: 508.453 KB)
4-element Array{Float64,1}:
 5.00001e11
 5.00001e11
 5.00001e11
 5.00001e11
