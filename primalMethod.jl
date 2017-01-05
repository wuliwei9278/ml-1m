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

function obtain_g(U, V, X, d1, d2, lambda, rows, vals)
	g = lambda * V;
	m = spzeros(d2,d1);
	for i = 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		ui = U[:, i]
		for j in d2_bar
			m[j,i] = dot(ui,V[:,j])
		end

		vals_d2_bar = vals[tmp];
		len = size(d2_bar)[1];
		t = spzeros(1, len);
		for j in 1:(len - 1)
			J = d2_bar[j];
			for k in (j + 1):len
				K = d2_bar[k];
				if vals_d2_bar[j] > vals_d2_bar[k]
					t = helper(m, t, i, j, k, J, K)
				elseif vals_d2_bar[k] > vals_d2_bar[j]
					t = helper(m, t, i, k, j, K, J)
				end
			end
		end
		
		for j in 1:len
			J = d2_bar[j]
			g[:,J] += ui * t[j]
		end
	end
	return g, m
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
	err = norm(rr) * 10.0^-3
	for k in 1:10
		#println(k)
		Hp = compute_Ha(p, m, U, X, r, d1, d2, lambda, rows, vals)
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
				mask = y_ipq * (m[p, i] - m[q, i])
				if mask >= 1.0
					continue
				else
					res += (1.0 - mask) ^ 2
				end
			end
		end
	end
	return res
end

function update_V(U, V, X, r, d1, d2, lambda, rows, vals, stepsize)
	g,m = obtain_g(U, V, X, d1, d2, lambda, rows, vals)
	g = vec(g)
	delta = solve_delta(g, m, U, X, r, d1, d2, lambda, rows, vals)
	delta = reshape(delta, size(V))
	obj = objective(m, U, V, X, d1, lambda, rows, vals)
	println("updating V: objective function value:", obj)
	V -= stepsize * delta
	return V
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


function obtain_g_u(A, D, V_bar, ui, lambda)
	tmp = A' * (V_bar' * ui) 
	tmp -= ones(size(A)[2])
	tmp = D * tmp
	tmp = A * tmp
	tmp = 2 * V_bar * tmp
	tmp += lambda * ui
	return tmp
end

function obtain_Hs(s, A, D, V_bar, lambda)
	tmp = A' * (V_bar' * s)
	tmp = D * tmp
	tmp = A * tmp
	tmp = 2 * V_bar * tmp
	tmp += lambda * s
	return tmp
end

function solve_delta_u(g, A, D, V_bar, lambda)
	# use linear conjugate grad descent
	delta = zeros(size(g)[1])
	rr = -g
	p = -rr
	err = norm(rr) * 10.0^-3
	for k in 1:10
		#println(k)
		Hp = obtain_Hs(p, A, D, V_bar, lambda)
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

function update_u(i, ui, V, X, r, d2, lambda, rows, vals, stepsize)
	A, D, V_bar, m, c = helper2(i, ui, V, X, r, d2, rows, vals)
	if c == 0.0
		return ui, -1.0
	end
	g = obtain_g_u(A, D, V_bar, ui, lambda)
	delta = solve_delta_u(g, A, D, V_bar, lambda)
	obj = objective_u(i, m, X, lambda, rows, vals, ui)
	println("Updating u: ", i, " objective function value:", obj)
	ui -= stepsize * delta
	return ui, obj
end


function update_U(U, V, X, r, d1, d2, lambda, rows, vals, stepsize)
	for i in 1:d1
		ui = U[:, i]
		prev = 0
		for k in 1:20
			println("k value: ", k)
			ui, obj = update_u(i, ui, V, X, r, d2, lambda, rows, vals, stepsize)
			if obj == -1.0
				break
			end
			if k == 1
				prev = obj
			else
				if abs(prev - obj) < 10.0 ^ -5 || (prev - obj) / prev < 10.0 ^ -1
					break
				end
				prev = obj
			end	
			println("prev value: ", prev)
		end	
		U[:, i] = ui
	end
	return U
end

function main()
	X = readdlm("MovieLens1m.csv", ',' , Int64);
	x = vec(X[:,1]) + 1; # userid starting from 0
	y = vec(X[:,2]) + 1; # same for movieid
	v = vec(X[:,3]);
	# userid; movieid
	n = 6040; m = 3952;
	X = sparse(x, y, v, n, m); # userid by movieid
	# julia column major 
	# now moveid by userid
	X = X'; 

	# too large to debug the algorithm, subset a small set: 500 by 750
	X = X[1:500, 1:750];
	rows = rowvals(X);
	vals = nonzeros(X);
	d2, d1 = size(X);
	r = 10; lambda = 5000;
	# initialize U, V
	U = 0.1*randn(r, d1);
	V = 0.1*randn(r, d2);
	stepsize = 1
	for iter in 1:5
		println("Outer iteration: ", iter)

		V = update_V(U, V, X, r, d1, d2, lambda, rows, vals, stepsize)

		U = update_U(U, V, X, r, d1, d2, lambda, rows, vals, stepsize)

		g,m = obtain_g(U, V, X, d1, d2, lambda, rows, vals)
		println(objective(m, U, V, X, d1, lambda, rows, vals))
	end
	return V, U
end