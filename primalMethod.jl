# julia

### Questions: 
#
#  1. compute_H_new(), should we have for loop: for j=1:len, for k=1:len ?  (you need to sum over all other elements for a j, see (7) in your writeup). 
#
#
#
####



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

function obtain_g_new(U, V, X, d1, d2, lambda, rows, vals, m)
	g = lambda * V;

#	ff = open("pppp", "w");
	for i = 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		vals_d2_bar = vals[tmp];
		len = size(d2_bar)[1];
		ui = U[:, i]

		mm = nonzeros(m[:,i])

		t = zeros(len);
		for j in 1:(len - 1)
#			J = d2_bar[j];
			for k in (j + 1):len
#				K = d2_bar[k];
				if vals_d2_bar[j] > vals_d2_bar[k]
#		println(ff, i, " ", J, " ", K);
					mask = mm[j] - mm[k];
					if mask < 1.0
						s_jk = 2.0 * (mask - 1.0)
						t[j] += s_jk
						t[k] -= s_jk
					end
				elseif vals_d2_bar[k] > vals_d2_bar[j]
#					println(ff, i, " ", K, " ", J);
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
			g[:,J] += ui * t[j]
		end
	end
#	close(ff)
	return g
end


function comp_m(U, V, X, d1, d2, rows, vals, cols)

	mvals = zeros(nnz(X))
	cc = 0
	for i = 1:d1
		tmp = nzrange(X,i)
		d2_bar = rows[tmp];
		ui = U[:,i]
		for j in d2_bar
			cc += 1
			mvals[cc] = dot(ui, V[:,j])
		end
	end
	return sparse(rows, cols, mvals, d2, d1);

#	m = spzeros(d2,d1);
#	for i = 1:d1
#		tmp = nzrange(X, i)
#		d2_bar = rows[tmp];
#		ui = U[:, i]
#		for j in d2_bar
#			m[j,i] = dot(ui,V[:,j])
#		end
#	end
#	return m
end


function compute_Ha_new(a, m, U, X, r, d1, d2, lambda, rows, vals)
	Ha = lambda * a
	for i in 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp]
		vals_d2_bar = vals[tmp]
		len = size(d2_bar)[1]

		b = zeros(len)
		ui = U[:,i]
		cc=0
		for q in d2_bar
			cc+=1
#			a_q = a[(q-1)*r+1:q*r]
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
			Ha[(p - 1) * r + 1 : p * r] += cpvals[j]*ui
		end

#		for j in 1:len
#			p = d2_bar[j];
#			c_p = 0.0
#			for k in 1:len
#				if vals_d2_bar[j] == vals_d2_bar[k]
#					continue
#				elseif vals_d2_bar[j] > vals_d2_bar[k]
#					y_ipq = 1.0
#				elseif vals_d2_bar[k] > vals_d2_bar[j]
#					y_ipq = -1.0
#				end
#				mask = y_ipq * (mm[j] - mm[k])
#				if mask < 1.0
#					c_p += 2.0 * (b[j] - b[k])
#				end
#			end
#			Ha[(p - 1) * r + 1 : p * r] += c_p*ui
#		end
	end
	return Ha
end	


function compute_Ha(a, m, U, X, r, d1, d2, lambda, rows, vals)
	Ha = lambda * a
	for i in 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp]
		b = spzeros(1,d2)
		ui = U[:,i]
		for q in d2_bar
			a_q = a[(q-1)*r+1:q*r]
			b[1,q] = dot(ui, a_q)
		end

		vals_d2_bar = vals[tmp]
		len = size(d2_bar)[1]
		for j in 1:(len - 1)
			p = d2_bar[j];
			c_p = 0.0
			for k in (j + 1):len
				q = d2_bar[k]
				if vals_d2_bar[j] == vals_d2_bar[k]
					continue
				elseif vals_d2_bar[j] > vals_d2_bar[k]
					y_ipq = 1.0
				elseif vals_d2_bar[k] > vals_d2_bar[j]
					y_ipq = -1.0
				end
				mask = y_ipq * (m[p, i] - m[q, i])
				if mask >= 1.0
					continue
				else
					s_pq = 2.0
					c_p += s_pq * (b[1,p] - b[1,q])
				end
			end
			Ha[(p - 1) * r + 1 : p * r] += ui * c_p
		end
	end
	return Ha
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
	res = lambda / 2 * (vecnorm(U) ^ 2 +vecnorm(V) ^ 2)
	for i in 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		vals_d2_bar = vals[tmp];
		len = size(d2_bar)[1];
		mm = nonzeros(m[:,i])

		for j in 1:(len - 1)
#			p = d2_bar[j];
			for k in (j + 1):len
#				q = d2_bar[k]
				if vals_d2_bar[j] == vals_d2_bar[k]
					continue
				elseif vals_d2_bar[j] > vals_d2_bar[k]
					y_ipq = 1.0
				elseif vals_d2_bar[k] > vals_d2_bar[j]
					y_ipq = -1.0
				end
				mask = y_ipq * (mm[j]-mm[k])
				if mask < 1.0
					res += (1.0 - mask) ^ 2
				end
			end
		end
	end
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

	VV = zeros(r,d2);
	for ii=1:r
		for jj = 1:d2
			VV[ii,jj] = V[ii,jj]
		end
	end

	return VV, m, new_obj
end

# Fix V, update U


function helper2(i, ui, V, X, r, d2, rows, vals)
	tmp = nzrange(X, i)
	d2_bar = rows[tmp];
	m = spzeros(1, d2)
	# need to get new m for updated V
	for j in d2_bar
		m[1,j] = dot(ui,V[:,j])
	end

	vals_d2_bar = vals[tmp];
	len = size(d2_bar)[1];
	num = round(Int64, len*(len-1)/2)
	D = zeros(num)
	A = spzeros(len, num)
	V_bar = zeros(r, len)
	c = 0
	
	for j in 1:len
		p = d2_bar[j];
		V_bar[:,j] = V[:,p]
		for k in (j + 1):len
			q = d2_bar[k]
			if vals_d2_bar[j] == vals_d2_bar[k]
				continue
			elseif vals_d2_bar[j] > vals_d2_bar[k]
				y_ipq = 1.0
				c += 1
				A[j, c] = 1.0; A[k, c] = -1.0
			elseif vals_d2_bar[k] > vals_d2_bar[j]
				y_ipq = -1.0
				c += 1
				A[j, c] = -1.0; A[k, c] = 1.0
			end
			mask = y_ipq * (m[1, p] - m[1, q])
			#println(mask)
			if mask >= 1.0
				continue
			else
				D[c] = 1.0
			end
		end
	end

	D = D[1:c]; A = A[:,1:c]
	D = spdiagm(D)
	return A, D, V_bar, m, c
end

function solve_delta_u(g, D, lambda, i, V, r, d2, vals, X, rows)
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

function compute_mm(i, ui, V, X, r, d2, rows, vals)
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


function objective_u_new(i, X, lambda, rows, vals, ui, mm)
	res = lambda / 2 * (vecnorm(ui) ^ 2)
	tmp = nzrange(X, i)
	d2_bar = rows[tmp];
	vals_d2_bar = vals[tmp];
	len = size(d2_bar)[1];
	for j in 1:(len - 1)
#		p = d2_bar[j];
		for k in (j + 1):len
#			q = d2_bar[k]
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


function objective_u(i, m, X, lambda, rows, vals, ui)
	res = lambda / 2 * (vecnorm(ui) ^ 2)
	tmp = nzrange(X, i)
	d2_bar = rows[tmp];
	vals_d2_bar = vals[tmp];
	len = size(d2_bar)[1];
	for j in 1:(len - 1)
		p = d2_bar[j];
		for k in (j + 1):len
			q = d2_bar[k]
			if vals_d2_bar[j] == vals_d2_bar[k]
				continue
			elseif vals_d2_bar[j] > vals_d2_bar[k]
				y_ipq = 1.0
			elseif vals_d2_bar[k] > vals_d2_bar[j]
				y_ipq = -1.0
			end
			mask = y_ipq * (m[1, p] - m[1, q])
			if mask >= 1.0
				continue
			else
				res += (1.0 - mask) ^ 2
			end
		end
	end
	return res
end

function obtain_g_u_new(i, ui, V, X, r, d2, rows, vals, lambda, mm)
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

function obtain_Hs_new(i, V, X, r, d2, rows, vals, lambda, D, s)
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



function update_u(i, ui, V, X, r, d2, lambda, rows, vals, stepsize, mm)

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
	total_obj_new = lambda/2*(vecnorm(V)^2)
	obj_new = 0

	for i in 1:d1
		ui = U[:, i]
# The reason of adding of commented codes before is to consider the case there are no comparisions for a particular user.
#		prev = 0
		mm = nonzeros(m[:,i]);
		for k in 1:1
			ui, obj_new, mm = update_u(i, ui, V, X, r, d2, lambda, rows, vals, stepsize, mm);
#			if obj == -1.0
#				break
#			end
#			if k == 1
#				prev = obj
#			else
#				if abs(prev - obj) < 10.0 ^ -5 || (prev - obj) / prev < 10.0 ^ -1
#					break
#				end
#				prev = obj
#			end	
#		println("prev value: ", prev)
		end	
		total_obj_new += obj_new
		U[:, i] = ui
	end
#	println(" OBJNEW: ", total_obj_new)
	return U, total_obj_new
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


function compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t)
	sum_error = 0.
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
	end
	return sum_error / d1
end

function computer_NDCG(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	ndcg_sum = 0.
	for i = 1:d1
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
		ndcg_sum += dcg / dcg_max
	end
	return ndcg_sum / d1
end




#X = readdlm("MovieLens1m.csv", ',' , Int64);
#x = vec(X[:,1]) + 1; # userid starting from 0
#y = vec(X[:,2]) + 1; # same for movieid
#v = vec(X[:,3]);

X = readdlm("ml1m_train_ratings.csv", ',' , Int64);
x = vec(X[:,1]);
y = vec(X[:,2]);
v = vec(X[:,3]);
Y = readdlm("ml1m_test_ratings.csv", ',' , Int64);
xx = vec(Y[:,1]);
yy = vec(Y[:,2]);
vv = vec(Y[:,3]);

#main(x, y, v);
#main(x, y, v, xx, yy, vv)

function main(x, y, v, xx, yy, vv)
	# userid; movieid
	# n = 6040; msize = 3952;
	# depending on the size of X, read n_users and n_items from python output
	n = 1496; msize = 3952; 
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
	ndcg_k = 10;
	# initialize U, V
	#srand(1234)
	U = 0.1*randn(r, d1);
	V = 0.1*randn(r, d2);
	stepsize = 1

	totaltime = 0.00000;
	println("iter time objective_function pairwise_error NDCG")
	pairwise_error, ndcg = compute_pairwise_error_ndcg(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	m = comp_m(U, V, X, d1, d2, rows, vals, cols)
	nowobj = objective(m, U, V, X, d1, lambda, rows, vals)
	println(" ", 0, " ", totaltime, " ", nowobj, " ", pairwise_error, " ", ndcg, ";")

	for iter in 1:20
		tic();
#	println("Outer iteration: ", iter)

#@time V, m, nowobj  = update_V(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, cols)
	
#@time U, nowobj = update_U(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, m)
		
		V, m, nowobj  = update_V(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, cols)
		U, nowobj = update_U(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, m)
	 	
		totaltime += toq();

		# need to add codes for computing pairwise error and NDCG

	 	#pairwise_error = compute_pairwise_error(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t)
	 	#ndcg = computer_NDCG(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)

	 	pairwise_error, ndcg = compute_pairwise_error_ndcg(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
		println(" ", iter, " ", totaltime, " ", nowobj, " ", pairwise_error, " ", ndcg, ";")

	end
#	return V, U
end

