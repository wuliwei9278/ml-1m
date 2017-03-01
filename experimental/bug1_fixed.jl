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

@everywhere function copy!(g, a)
	for j in 1:size(a)[2]
		for i in 1:size(a)[1]
			g[i,j] += a[i,j]
		end
	end
end

@everywhere function chunk_g!(g, U, V, X, d1, d2, lambda, rows, vals, m, irange)
	# display so we can see what's happening
    #@show (irange)  
    a = zeros(size(V))
    for i in irange
        tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		vals_d2_bar = vals[tmp];
		len = size(d2_bar)[1];
		ui = U[:, i]

		mm = nonzeros(m[:,i])

		t = zeros(len);
		for j in 1:(len - 1)
			for k in (j + 1):len
				if vals_d2_bar[j] > vals_d2_bar[k]
					mask = mm[j] - mm[k];
					if mask < 1.0
						s_jk = 2.0 * (mask - 1.0)
						t[j] += s_jk
						t[k] -= s_jk
					end
				elseif vals_d2_bar[k] > vals_d2_bar[j]
					mask = mm[k]-mm[j];
					if mask < 1.0
						s_jk = 2.0 * (mask - 1.0)
						t[k] += s_jk
						t[j] -= s_jk
					end
				end
			end
		end
		
		for j in 1:len
			J = d2_bar[j]
			a[:,J] += ui * t[j]
		end
    end
    copy!(g, a)
    return g
end

@everywhere shared_chunk_g!(g, U, V, X, d1, d2, lambda, rows, vals, m) = chunk_g!(g, U, V, X, d1, d2, lambda, rows, vals, m, myrange(d1))


function obtain_g_new(U, V, X, d1, d2, lambda, rows, vals, m)
	g = SharedArray(Float64, size(V))
    @sync begin
        for p in 1:nprocs()
            @async remotecall_wait(p, shared_chunk_g!, g, U, V, X, d1, d2, lambda, rows, vals, m)
        end
    end
    g + lambda * V
end

@everywhere function copy2!(Ha, d)
	for i in 1:size(d)[1]
		Ha[i] += d[i]
	end
end

@everywhere function chunk_Ha!(Ha, a, m, U, X, r, d1, d2, lambda, rows, vals, irange)
	# display so we can see what's happening
    #@show (irange)  
    d = zeros(size(a))
    for i in irange
        tmp = nzrange(X, i)
		d2_bar = rows[tmp]
		vals_d2_bar = vals[tmp]
		len = size(d2_bar)[1]

		b = zeros(len)
		ui = U[:,i]
		cc=0
		for q in d2_bar
			cc+=1
			b[cc] = dot(ui, a[(q-1)*r+1:q*r])
		end

		mm = nonzeros(m[:,i])

		cpvals = zeros(len)
		for j in 1:(len - 1)
			jval = vals_d2_bar[j]
			for k in (j + 1):len
				kval = vals_d2_bar[k]
				if jval == kval
					continue
				elseif jval > kval
					y_ipq = 1.0
				else 
					y_ipq = -1.0
				end
				mask = y_ipq * (mm[j] - mm[k])
				if mask < 1.0
					ddd = 2.0*(b[j]-b[k])
					cpvals[j] += ddd
					cpvals[k] -= ddd
				end
			end
		end
		for j in 1:len
			p = d2_bar[j]
			d[(p - 1) * r + 1 : p * r] += cpvals[j]*ui
		end
    end
    copy2!(Ha, d)
    return Ha
end

@everywhere shared_chunk_Ha!(Ha, a, m, U, X, r, d1, d2, lambda, rows, vals) = chunk_Ha!(Ha, a, m, U, X, r, d1, d2, lambda, rows, vals, myrange(d1))

function compute_Ha_new(a, m, U, X, r, d1, d2, lambda, rows, vals)
	Ha = SharedArray(Float64, size(a))
    @sync begin
        for p in 1:nprocs()
            @async remotecall_wait(p, shared_chunk_Ha!, Ha, a, m, U, X, r, d1, d2, lambda, rows, vals)
        end
    end
    Ha + lambda * a
end


# now 4 cores
julia> main(x, y, v, xx, yy, vv)
iter time objective_function pairwise_error NDCG
[0, 0.0, 3.492397234471353e7, 0.3510205089577718, 0.4753335862187586],
  1.895354 seconds (2.29 M allocations: 677.565 MB, 6.15% gc time)
  0.735858 seconds (13.94 k allocations: 1.195 MB)
[1, 2.631570828, 2.065028912258863e7, 0.23639373637664418, 0.7336591120290276],
  2.429199 seconds (2.31 M allocations: 769.580 MB, 2.37% gc time)
  0.868129 seconds (13.96 k allocations: 1.212 MB)
[2, 5.929527539, 1.7919968349510193e7, 0.19271221653854162, 0.7780543218090921],
  2.959689 seconds (2.32 M allocations: 800.282 MB, 4.70% gc time)
  0.936028 seconds (13.94 k allocations: 1.227 MB)
[3, 9.825584986, 1.770736360514091e7, 0.18473524373363082, 0.7910130941548634],
  3.534186 seconds (2.32 M allocations: 830.981 MB, 2.21% gc time)
  1.200931 seconds (13.95 k allocations: 1.211 MB)
[4, 14.561187160000001, 1.7663140251798525e7, 0.18201433596305322, 0.796329763274301],
  3.348035 seconds (2.32 M allocations: 830.990 MB, 3.29% gc time)
  0.968449 seconds (13.94 k allocations: 1.227 MB)
[5, 18.878110928, 1.7649781189849637e7, 0.18098679653483146, 0.7981006428110772],
  2.865096 seconds (2.32 M allocations: 800.283 MB, 2.14% gc time)
  0.912097 seconds (13.95 k allocations: 1.227 MB)

Questions: 8 cores does not improve that much?! 
julia> main(x, y, v, xx, yy, vv)
iter time objective_function pairwise_error NDCG
[0, 0.0, 3.4923972344713524e7, 0.3510205089577718, 0.4753335862187586],
  1.508429 seconds (2.37 M allocations: 683.837 MB, 4.32% gc time)
  0.562652 seconds (29.53 k allocations: 2.548 MB, 0.53% gc time)
[1, 2.071424821, 2.065018647888106e7, 0.23643604844121305, 0.733801658808334],
  2.347447 seconds (2.41 M allocations: 778.241 MB, 6.73% gc time)
  0.699278 seconds (29.54 k allocations: 2.533 MB)
[2, 5.118535867, 1.7919936938246097e7, 0.19270179046595617, 0.7781138146180607],
  2.455871 seconds (2.43 M allocations: 809.688 MB, 2.91% gc time)
  0.812642 seconds (29.55 k allocations: 2.549 MB, 0.51% gc time)
[3, 8.387437163000001, 1.7707365328013077e7, 0.18474280750564767, 0.7910377289550884],
  3.198376 seconds (2.45 M allocations: 841.277 MB, 5.13% gc time)
  0.802887 seconds (29.53 k allocations: 2.533 MB)
[4, 12.388998292, 1.76631486699426e7, 0.18201994346977354, 0.7962951721315066],
  2.745884 seconds (2.45 M allocations: 841.192 MB, 2.87% gc time)
  0.749200 seconds (29.54 k allocations: 2.533 MB)
[5, 15.884585339000001, 1.7649786063768107e7, 0.1809870471708711, 0.7980861015072485],
  2.800363 seconds (2.43 M allocations: 809.740 MB, 5.60% gc time)
  0.767100 seconds (29.54 k allocations: 2.517 MB)

# before 4 cores
julia> main(x, y, v, xx, yy, vv)
iter time objective_function pairwise_error NDCG
[0, 0.0, 3.492397234471353e7, 0.3510205089577718, 0.4753335862187586],
  3.761987 seconds (9.10 M allocations: 8.884 GB, 17.65% gc time)
  0.741054 seconds (13.95 k allocations: 1.196 MB)
[1, 4.503359734, 2.960810245444077e7, 0.3505945679165355, 0.47194744724999294],
  8.334163 seconds (29.14 M allocations: 13.239 GB, 14.67% gc time)
  0.700449 seconds (13.95 k allocations: 1.196 MB)
[2, 13.538423657, 2.9607957575345986e7, 0.35052273209325885, 0.4725454810407132],
  8.158798 seconds (29.14 M allocations: 13.239 GB, 15.21% gc time)
  0.786712 seconds (13.96 k allocations: 1.196 MB)
[3, 22.484396025, 2.960795756834907e7, 0.35052137897410623, 0.4725489209604168],
  8.369406 seconds (29.14 M allocations: 13.239 GB, 15.31% gc time)
  0.844123 seconds (13.96 k allocations: 1.212 MB)
[4, 31.698274847, 2.960795756901582e7, 0.35052126707004627, 0.4725489209604168],
  8.292371 seconds (29.14 M allocations: 13.239 GB, 15.37% gc time)
  0.357423 seconds (13.96 k allocations: 1.212 MB)
[5, 40.348538767, 2.9607957569683086e7, 0.3505212655933848, 0.4725489209604168],
  8.109621 seconds (29.14 M allocations: 13.239 GB, 14.90% gc time)
  0.565849 seconds (13.95 k allocations: 1.212 MB)
[6, 49.024444749, 2.960795757035034e7, 0.3505212655933848, 0.4725489209604168],
  8.120349 seconds (29.14 M allocations: 13.239 GB, 14.94% gc time)
  0.634642 seconds (13.95 k allocations: 1.212 MB)
[7, 57.779789619999995, 2.96079575710176e7, 0.3505212655933848, 0.4725489209604168],
  8.468281 seconds (29.14 M allocations: 13.239 GB, 15.59% gc time)
  0.632011 seconds (13.97 k allocations: 1.213 MB)

# before 8 cores
julia> main(x, y, v, xx, yy, vv)
iter time objective_function pairwise_error NDCG
[0, 0.0, 3.4923972344713524e7, 0.3510205089577718, 0.4753335862187586],
  3.045131 seconds (9.13 M allocations: 4.482 GB, 13.08% gc time)
  0.532930 seconds (29.54 k allocations: 2.486 MB)
[1, 3.578344011, 2.065018648247156e7, 0.23643604844121305, 0.733801658808334],
  5.060367 seconds (13.68 M allocations: 7.612 GB, 13.64% gc time)
  0.732674 seconds (29.57 k allocations: 2.534 MB, 0.46% gc time)
[2, 9.371849556999999, 1.7919937083181817e7, 0.1927016951718535, 0.7781138146180607],
  5.765401 seconds (15.19 M allocations: 8.656 GB, 13.93% gc time)
  0.784837 seconds (29.55 k allocations: 2.549 MB)
[3, 15.922488824999999, 1.770736560111908e7, 0.18474258794462156, 0.7910377289550884],
  6.417435 seconds (16.71 M allocations: 9.699 GB, 13.25% gc time)
  0.763956 seconds (29.57 k allocations: 2.550 MB)
[4, 23.104347925, 1.766314870528199e7, 0.18201973339141664, 0.796294944993741],
  6.405463 seconds (16.71 M allocations: 9.699 GB, 14.05% gc time)
  0.773094 seconds (29.55 k allocations: 2.534 MB)
[5, 30.283240657, 1.764978595047793e7, 0.1809857839424875, 0.7980915688394775],
  5.735589 seconds (15.19 M allocations: 8.656 GB, 13.28% gc time)
  0.763048 seconds (29.53 k allocations: 2.549 MB)
[6, 36.782397035, 1.7644575966865066e7, 0.18045180573422978, 0.7994476981028341],
  5.657927 seconds (15.19 M allocations: 8.656 GB, 13.75% gc time)
  0.720771 seconds (29.55 k allocations: 2.565 MB)
[7, 43.16145403, 1.764208463815182e7, 0.1802438940370599, 0.7994307352179486],
  4.998919 seconds (13.68 M allocations: 7.612 GB, 13.71% gc time)
  0.743225 seconds (29.54 k allocations: 2.549 MB)
  

