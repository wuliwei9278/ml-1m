# julia
# Fix U, update V

function helper(m, t, i, j, k, J, K)
	mask = m[J, i] - m[K, i]
	if mask >= 1.0
		return t
	else
		s_jk = 2.0 * (mask - 1.0)
		t[j] += s_jk
		t[k] -= s_jk
	end
	return t
end

function comp_m(U, V, X, d1, d2, rows, vals, cols)

	mvals = zeros(nnz(X))
	cc = 0
	for i = 1:d1
		tmp = nzrange(X,i)
		d2_bar = rows[tmp]
		ui = U[:,i]
		for j in d2_bar
			cc += 1
			mvals[cc] = dot(ui, V[:,j])
		end
	end
	return sparse(rows, cols, mvals, d2, d1)
end

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

@everywhere function update(g, a)
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
    update(g, a)
    return g
end

@everywhere shared_chunk_g!(g, U, V, X, d1, d2, lambda, rows, vals, m) = chunk_g!(g, U, V, X, d1, d2, lambda, rows, vals, m, myrange(d1))


function obtain_g_new(U, V, X, d1, d2, lambda, rows, vals, m)
	g = SharedArray(Float64, size(V))
    @sync begin
        for p in 1:nprocs()
            @async remotecall_wait(shared_chunk_g!, p, g, U, V, X, d1, d2, lambda, rows, vals, m)
        end
    end
    g + lambda * V
end

@everywhere function update2(Ha, d)
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
    update2(Ha, d)
    return Ha
end

@everywhere shared_chunk_Ha!(Ha, a, m, U, X, r, d1, d2, lambda, rows, vals) = chunk_Ha!(Ha, a, m, U, X, r, d1, d2, lambda, rows, vals, myrange(d1))

function compute_Ha_new(a, m, U, X, r, d1, d2, lambda, rows, vals)
	Ha = SharedArray(Float64, size(a))
    @sync begin
        for p in 1:nprocs()
            @async remotecall_wait(shared_chunk_Ha!, p, Ha, a, m, U, X, r, d1, d2, lambda, rows, vals)
        end
    end
    Ha + lambda * a
end


function solve_delta(g, m, U, X, r, d1, d2, lambda, rows, vals)
	# use linear conjugate grad descent
	delta = zeros(size(g)[1])
	rr = -g
	p = -rr
	err = norm(rr) * 10.0^-2
	for k in 1:10
		#Hp = compute_Ha(p, m, U, X, r, d1, d2, lambda, rows, vals)
		Hp = compute_Ha_new(p, m, U, X, r, d1, d2, lambda, rows, vals)
		alpha = -dot(rr, p) / dot(p, Hp)
		delta += alpha * p
		rr += alpha * Hp
		if norm(rr) < err
			break
		end
		#println(norm(rr))
		b = dot(rr, Hp) / dot(p, Hp)
		p = -rr + b * p
	end
	return delta
end


function objective(m, U, V, X, d1, lambda, rows, vals)
	res = 0.
	res = @parallel (+) for i in 1:d1
		s = 0.
		tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		vals_d2_bar = vals[tmp];
		len = size(d2_bar)[1];
		mm = nonzeros(m[:,i])

		for j in 1:(len - 1)
			for k in (j + 1):len
				if vals_d2_bar[j] == vals_d2_bar[k]
					continue
				elseif vals_d2_bar[j] > vals_d2_bar[k]
					y_ipq = 1.0
				elseif vals_d2_bar[k] > vals_d2_bar[j]
					y_ipq = -1.0
				end
				mask = y_ipq * (mm[j]-mm[k])
				if mask < 1.0
					s += (1.0 - mask) ^ 2
				end
			end
		end
		s
	end
	res += lambda / 2 * (vecnorm(U) ^ 2 +vecnorm(V) ^ 2)
	return res
end

function update_V(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, cols)
	m = comp_m(U, V, X, d1, d2, rows, vals, cols);
  	g = obtain_g_new(U, V, X, d1, d2, lambda, rows, vals,m)
	delta = solve_delta(vec(g), m, U, X, r, d1, d2, lambda, rows, vals)
	delta = reshape(delta, size(V))
	prev_obj = objective(m, U, V, X, d1, lambda, rows, vals)

	Vold = V;
	s = stepsize
	new_obj=0.0
	for iter=1:20
		V = Vold - s * delta
		m = comp_m(U, V, X, d1, d2, rows, vals, cols);
		new_obj = objective(m, U, V, X, d1, lambda, rows, vals)
		#println("Line Search iter ", iter, " Prev Obj ", prev_obj, " New Obj ", new_obj)
		if (new_obj < prev_obj)
			break
		else
			s/=2
		end
	end

	return V, m, new_obj
end

# Fix V, update U

@everywhere function solve_delta_u(g, D, lambda, i, V, r, d2, vals, X, rows)
	# use linear conjugate grad descent
	delta = zeros(size(g)[1])
	rr = -g
	p = -rr
	err = norm(rr) * 10.0^-2
	for k in 1:10
		Hp = obtain_Hs_new(i, V, X, r, d2, rows, vals, lambda, D, p);
		alpha = -dot(rr, p) / dot(p, Hp)
		delta += alpha * p
		rr += alpha * Hp
		if norm(rr) < err
			break
		end
		#println(norm(rr))
		b = dot(rr, Hp) / dot(p, Hp)
		p = -rr + b * p
	end
	return delta
end

@everywhere function compute_mm(i, ui, V, X, r, d2, rows, vals)
	tmp = nzrange(X, i)
	d2_bar = rows[tmp];
	len = size(d2_bar)[1];
	mm = zeros(len);
	c=0;
	for j in d2_bar
		c+=1
		mm[c] = dot(ui,V[:,j])
	end
	return mm
end


@everywhere function objective_u_new(i, X, lambda, rows, vals, ui, mm)
	res = lambda / 2 * (vecnorm(ui) ^ 2)
	tmp = nzrange(X, i)
	d2_bar = rows[tmp];
	vals_d2_bar = vals[tmp];
	len = size(d2_bar)[1];
	for j in 1:(len - 1)
		for k in (j + 1):len
			if vals_d2_bar[j] == vals_d2_bar[k]
				continue
			elseif vals_d2_bar[j] > vals_d2_bar[k]
				y_ipq = 1.0
			else
				y_ipq = -1.0
			end
			mask = y_ipq * (mm[j] - mm[k])
			if mask < 1.0
				res += (1.0 - mask) ^ 2
			end
		end
	end
	return res
end



@everywhere function obtain_g_u_new(i, ui, V, X, r, d2, rows, vals, lambda, mm)
	tmp = nzrange(X, i)
	len = size(tmp)[1];
	if len==0
		return g, g, 0
	end
	d2_bar = rows[tmp];

	vals_d2_bar = vals[tmp];
	num = round(Int64, len*(len-1)/2)
	D = zeros(Int, num)

	g = zeros(r)
	tmp_vals = zeros(len)

	c = 0
	for j = 1:len-1
		for k = (j + 1):len
			if vals_d2_bar[j] == vals_d2_bar[k]
				continue
			elseif vals_d2_bar[j] > vals_d2_bar[k]
				y_ipq = 1.0
			else
				y_ipq = -1.0
			end
			c+=1
				mask = y_ipq * (mm[j]-mm[k])
			if mask < 1.0
						D[c] = 1.0;
						aaa = 2*(1-mask)*y_ipq;
						tmp_vals[j] -= aaa;
						tmp_vals[k] += aaa;
			end
		end
	end

	for j in 1:len
		p = d2_bar[j];
		g += tmp_vals[j]*V[:,p];
	end

	g += lambda * ui
	return g, D, c
end

@everywhere function obtain_Hs_new(i, V, X, r, d2, rows, vals, lambda, D, s)
	tmp = nzrange(X, i)
	d2_bar = rows[tmp];
	vals_d2_bar = vals[tmp];
	len = size(d2_bar)[1];

	m = zeros(len)
	# need to get new m for updated V
	c=0;
	for j in d2_bar
		c+=1;
		m[c] = dot(s,V[:,j])
	end


	g = zeros(r)
	tmp_vals = zeros(len)

	c = 0
	for j in 1:len
		for k in (j + 1):len
			if vals_d2_bar[j] == vals_d2_bar[k]
				continue
			end
			c+=1
			mask = m[j] - m[k];
			if D[c] > 0.0
						aaa = 2.0*mask;
						tmp_vals[j] += aaa;
						tmp_vals[k] -= aaa;
			end
		end
	end

	for j in 1:len
		p = d2_bar[j];
		g += tmp_vals[j]*V[:,p];
	end

	g += lambda * s
	return g
end



@everywhere function update_u(i, ui, V, X, r, d2, lambda, rows, vals, stepsize, mm)

	new_obj = 0
	
	g, D,c = obtain_g_u_new(i, ui, V, X, r, d2, rows, vals, lambda, mm);

	prev_obj = objective_u_new(i, X, lambda, rows, vals, ui, mm)
	if (c == 0.0) || (norm(g)<1e-4)
		return ui, prev_obj, mm
	end
	delta = solve_delta_u(g, D, lambda, i, V, r, d2, vals, X, rows)

	s = stepsize;
	for inneriter=1:20
		ui_new = ui - s * delta;
		mm_new = compute_mm(i, ui_new, V, X, r, d2, rows, vals);
		new_obj = objective_u_new(i, X, lambda, rows, vals, ui_new, mm_new);
		if ( new_obj < prev_obj)
			break;
		else
			s/=2.0;
		end
	end
	mm = mm_new;
	ui = ui_new;
	mm_new = 0;
	ui_new = 0;
	D = 0;
	g = 0;
	return ui, new_obj, mm
end


function update_U(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, m)
	obj_new = 0.
	obj_new = @parallel (+) for i in 1:d1
		ui = U[:, i]
		mm = nonzeros(m[:,i]);
		for k in 1:1
			ui, obj_new, mm = update_u(i, ui, V, X, r, d2, lambda, rows, vals, stepsize, mm);
		end	
		U[:, i] = ui
		obj_new
	end
	obj_new += lambda/2*(vecnorm(V)^2)
	return U, obj_new
end



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
    update2(res, a)
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




function compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t)
	sum_error = 0.
	sum_error = @parallel (+) for i = 1:d1
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
		#sum_error += error_this / n_comps_this
		error_this / n_comps_this
	end
	return sum_error / d1
end

function compute_NDCG(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	ndcg_sum = 0.
	ndcg_sum = @parallel (+) for i = 1:d1
		tmp = nzrange(Y, i)
		d2_bar = rows_t[tmp]
		vals_d2_bar = vals_t[tmp]
		ui = U[:, i]
		len = size(d2_bar)[1]
		score = zeros(len)
		for j = 1:len
			J = d2_bar[j];
			vj = V[:, J]
			score[j] = dot(ui,vj)
		end
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
		#ndcg_sum += dcg / dcg_max
		dcg / dcg_max
	end
	return ndcg_sum / d1
end







# command to run julia program after include this file
# main("data/ml1m_train_ratings.csv", "data/ml1m_test_ratings.csv", 100, 5000)
function main(train, test, r, lambda)
	X = readdlm(train, ',' , Int64);
	x = vec(X[:,1]);
	y = vec(X[:,2]);
	v = vec(X[:,3]);
	Y = readdlm(test, ',' , Int64);
	xx = vec(Y[:,1]);
	yy = vec(Y[:,2]);
	vv = vec(Y[:,3]);
	# userid; movieid
	# n = 6040; msize = 3952;
	# depending on the size of X, read n_users and n_items from python output
	n = max(maximum(x), maximum(xx)); msize = max(maximum(y), maximum(yy));
	#n = 1496; msize = 3952; 
	#n = 12851; msize = 65134
    #n = 221004; msize = 17771 # 417737 users ratings >= 20
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


	#r = 100; 
	#lambda = 5000; 
	# lambda = 7000 # works better for movielens10m data
	#lambda = 10000; # works better for netflix data
	ndcg_k = 10;
	# initialize U, V
	srand(1234)
	U = 0.1*randn(r, d1); V = 0.1*randn(r, d2);
	# U = 0.01*randn(r, d1); V = 0.01*randn(r, d2); # works better for netflix data
	U = convert(SharedArray, U)
	V = convert(SharedArray, V)
	stepsize = 1

	totaltime = 0.00000;
	println("iter time objective_function pairwise_error NDCG")
	
	pairwise_error, ndcg = compute_pairwise_error_ndcg(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	#pairwise_error = compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t)
	#ndcg = compute_NDCG(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)

	m = comp_m(U, V, X, d1, d2, rows, vals, cols)
	nowobj = objective(m, U, V, X, d1, lambda, rows, vals)
	println("[", 0, ", ", totaltime, ", ", nowobj, ", ", pairwise_error, ", ", ndcg, "],")

	for iter in 1:20
		tic();

 		V, m, nowobj  = update_V(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, cols)
	
 		U, nowobj = update_U(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, m)
		

		totaltime += toq();

		# need to add codes for computing pairwise error and NDCG

		#pairwise_error = compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t)
		#ndcg = compute_NDCG(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	 	pairwise_error, ndcg = compute_pairwise_error_ndcg(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
		println("[", iter, ", ", totaltime, ", ", nowobj, ", ", pairwise_error, ", ", ndcg, "],")

	end
#	return V, U
end
