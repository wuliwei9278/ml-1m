@everywhere function chunk_ndcg!(res, U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k, irange)
	# display so we can see what's happening
    #@show (irange)  
    a = zeros(2)
    for i in irange
        tmp = nzrange(Y, i)
		d2_bar = rows_t[tmp];
		vals_d2_bar = vals_t[tmp];
		ui = U[:, i]
		len = size(d2_bar)[1]
		score = zeros(len)
		for j = 1:len
			J = d2_bar[j];
			vj = V[:, J]
			score[j] = dot(ui,vj)
		end
		error_this = 0
		n_comps_this = 0
		for j in 1:(len - 1)
			jval = vals_d2_bar[j]
			for k in (j + 1):len
				kval = vals_d2_bar[k]
				if score[j] >= score[k] && jval < kval
					error_this += 1
				end
				if score[j] <= score[k] && jval > kval
					error_this += 1
				end
				n_comps_this += 1
			end
		end
		a[1] += error_this / n_comps_this


		p1 = sortperm(score, rev = true)
		p1 = p1[1:ndcg_k]
		M1 = vals_d2_bar[p1]
		p2 = sortperm(vals_d2_bar, rev = true)
		p2 = p2[1:ndcg_k]
		M2 = vals_d2_bar[p2]
		dcg = 0.; dcg_max = 0.
		for k = 1:ndcg_k
			dcg += (2 ^ M1[k] - 1) / log2(k + 1)
			dcg_max += (2 ^ M2[k] - 1) / log2(k + 1)
		end
		a[2] += dcg / dcg_max
    end
    copy2(res, a)
    return res
end

@everywhere shared_chunk_ndcg!(res, U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k) = chunk_ndcg!(res, U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k, myrange(d1))

function compute_pairwise_error_ndcg(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	res = SharedArray(Float64, 2)
    @sync begin
        for p in 1:nprocs()
            @async remotecall_wait(shared_chunk_ndcg!, p, res, U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
        end
    end
    return res[1] / d1, res[2] / d1 
end

