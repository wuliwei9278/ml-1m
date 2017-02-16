function compute_RMSE(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t)
	res = 0.0
	n = 0.0
	for i = 1:d1
		tmp = nzrange(Y, i)
		d2_bar = rows_t[tmp];
		vals_d2_bar = vals_t[tmp];
		ui = U[:, i]
		len = size(d2_bar)[1]
		for j = 1:len
			J = d2_bar[j];
			vj = V[:, J]
			res += (vals_d2_bar[j] - dot(ui,vj))^2
			n += 1.0
		end
	end
	return (res / n)^0.5
end

function compute_RMSE_train(U, V, X, r, d1, d2, rows, vals, cols)
	res = 0.0
	n = 0.0
	for i = 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		vals_d2_bar = vals[tmp];
		ui = U[:, i]
		len = size(d2_bar)[1]
		for j = 1:len
			J = d2_bar[j];
			vj = V[:, J]
			res += (vals_d2_bar[j] - dot(ui,vj))^2
			n += 1.0
		end
	end
	return (res / n)^0.5
end

function compute_pairwise_error_ndcg(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	sum_error = 0.; ndcg_sum = 0.;
	for i = 1:d1
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
		sum_error += error_this / n_comps_this


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
		ndcg_sum += dcg / dcg_max
	end
	return sum_error / d1, ndcg_sum / d1
end

function objective(U, V, X, d1, lambda, rows, vals)
	res = 0.0
	res = lambda * (vecnorm(U) ^ 2 + vecnorm(V) ^ 2)
	for i in 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		vals_d2_bar = vals[tmp];
		len = size(d2_bar)[1];
		for k in 1:len
			j = d2_bar[k]
			res += (vals_d2_bar[k] - dot(U[:,i], V[:,j]))^2
		end
	end
	return res
end

function update(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, cols)
	i = rand(1:d1)
	j = rand(1:d2)
	eij = X[j, i] - dot(U[:,i], V[:,j])
	ui = U[:,i] + stepsize * (eij * V[:,j] - lambda * U[:,i])
	vj = V[:,j] + stepsize * (eij * U[:,i] - lambda * V[:,j])
	U[:,i] = ui
	V[:,j] = vj
	return U, V
end

X = readdlm("ml1m_train_ratings.csv", ',' , Int64);
#X = readdlm("ml10m3_train_ratings.csv", ',' , Int64);
x = vec(X[:,1]);
y = vec(X[:,2]);
v = vec(X[:,3]);
Y = readdlm("ml1m_test_ratings.csv", ',' , Int64);
#Y = readdlm("ml10m3_test_ratings.csv", ',' , Int64);
xx = vec(Y[:,1]);
yy = vec(Y[:,2]);
vv = vec(Y[:,3]);

function main(x, y, v, xx, yy, vv)
	# userid; movieid
	# n = 6040; msize = 3952;
	# depending on the size of X, read n_users and n_items from python output
	n = max(maximum(x), maximum(xx)); msize = max(maximum(y), maximum(yy));
	#n = 1496; msize = 3952; 
	#n = 12851; msize = 65134
    #n = 221004; msize = 17771
	X = sparse(x, y, v, n, msize); # userid by movieid
	Y = sparse(xx, yy, vv, n, msize);
	# julia column major 
	# now moveid by userid
	X = X'; 
	Y = Y'; 

	# too large to debug the algorithm, subset a small set: 500 by 750
	#X = X[1:500, 1:750];
	#X = X[1:1000, 1:2000];
	rows = rowvals(X);
	vals = nonzeros(X);
	cols = zeros(Int, size(vals)[1]);

	d2, d1 = size(X);
	cc = 0;
	for i = 1:d1
		tmp = nzrange(X, i);
		nowlen = size(tmp)[1];
		for j = 1:nowlen
			cc += 1
			cols[cc] = i
		end
	end

	rows_t = rowvals(Y);
	vals_t = nonzeros(Y);
	cols_t = zeros(Int, size(vals_t)[1]);
	cc = 0;
	for i = 1:d1
		tmp = nzrange(Y, i);
		nowlen = size(tmp)[1];
		for j = 1:nowlen
			cc += 1
			cols_t[cc] = i
		end
	end


	r = 100; 
	lambda = 5000; 
	#lambda = 7000 # works better for movielens10m data
	#lambda = 10000; # works better for netflix data
	ndcg_k = 10;
	# initialize U, V
	srand(1234)
	U = 0.1*randn(r, d1); V = 0.1*randn(r, d2);
	# U = 0.01*randn(r, d1); V = 0.01*randn(r, d2); # works better for netflix data

	#stepsize = 0.0001
	stepsize = 0.0001
	totaltime = 0.00000;
	println("iter time objective_function pairwise_error NDCG RMSE")
	

	pairwise_error, ndcg = compute_pairwise_error_ndcg(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	
	nowobj = objective(U, V, X, d1, lambda, rows, vals)
	rmse = compute_RMSE(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t)
	rmse_tr = compute_RMSE_train(U, V, X, r, d1, d2, rows, vals, cols)
	println("[", 0, ", ", totaltime, ", ", nowobj, ", ", pairwise_error, ", ", ndcg, ", ", rmse, ", ", rmse_tr, "],")

	for iter in 1:10000000
		tic();

 		#V, m, nowobj  = update_V(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, cols)
	
 		#U, nowobj = update_U(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, m)
		
		U, V = update(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, cols)

		totaltime += toq();

		# need to add codes for computing pairwise error and NDCG

		#pairwise_error = compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t)
		#ndcg = compute_NDCG(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	 	if iter % 50000 == 0
	 		nowobj = objective(U, V, X, d1, lambda, rows, vals)
	 		pairwise_error, ndcg = compute_pairwise_error_ndcg(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	 		rmse = compute_RMSE(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t)
	 		rmse_tr = compute_RMSE_train(U, V, X, r, d1, d2, rows, vals, cols)
			println("[", iter, ", ", totaltime, ", ", nowobj, ", ", pairwise_error, ", ", ndcg, ", ", rmse_tr, ", ", rmse, "],")
	 	end
	 	

	end
#	return V, U
end
