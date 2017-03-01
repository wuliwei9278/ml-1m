function compute_precision(U, V, X, Y, d1, d2, rows, vals, rows_t)
	K = [1, 5, 10, 100] # K has to be increasing order
	precision = [0, 0, 0, 0]
	for i = 1:d1
		tmp = nzrange(Y, i)
		test = Set(rows_t[tmp])
		if isempty(test)
			continue
		end
		tmp = nzrange(X, i)
		vals_d2_bar = vals[tmp]
		# need to distinguish 1 and -1, only treating 1 as train, since -1 can contain test data
		tmp = tmp[vals_d2_bar .== 1]
		train = Set(rows[tmp])
		score = zeros(d2)
		ui = U[:, i]
		for j = 1:d2
			if j in train
				score[j] = -10e10
				continue
			end
			vj = V[:, j]
			score[j] = dot(ui,vj)
		end
		p = sortperm(score, rev = true)
		for c = 1: K[length(K)]
			j = p[c]
			if score[j] == -10e10
				break
			end
			if j in test
				for k in length(K):-1:1
					if c <= K[k]
						precision[k] += 1
					else:
						break
					end
				end
			end
		end
	end
	precision = precision./K/d1
	return precision[1], precision[2], precision[3], precision[4]
end
