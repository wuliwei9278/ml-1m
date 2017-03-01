# julia


# Fix U, update V

type infor 
	levels::Array{Float64,1}
	nlevels::Int
	perm_ind::Array{Int,1}
	mm_sorted::Array{Float64,1}
	vals_sorted::Array{Int,1}
	d2bar_sorted::Array{Int,1}
	countright::Array{Int,1}
end


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
	g1 = lambda * V;

	totalnum =0;
#	ff = open("pppp", "w");
	for i = 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		vals_d2_bar = vals[tmp];
		len = size(d2_bar)[1];
		ui = U[:, i]

		mm = nonzeros(m[:,i])

		if len>=0 
		levels = sort(unique(vals_d2_bar))
		nlevels = size(levels)[1]
		perm_ind = sortperm(mm)
		mm_sorted = mm[perm_ind]
		b_sorted = mm_sorted
		vals_sorted = vals_d2_bar[perm_ind]
		for j=1:len
			for k=1:nlevels
				if vals_sorted[j] == levels[k]
					vals_sorted[j] = k
					break
				end
			end
		end
		d2bar_sorted = d2_bar[perm_ind]
		allsum = zeros(nlevels)
		for j=1:len
			allsum[vals_sorted[j]] += b_sorted[j]
		end


		nowleft = 1
		nowright = 1
		countleft = zeros(Int, nlevels)
		countright = zeros(Int, nlevels)
		for j=1:len
			countright[vals_sorted[j]] +=1
		end

		nowleftsum = zeros(nlevels)
		nowrightsum = allsum
		cplist = zeros(len)
		for j = 1:len
			nowcut = mm_sorted[j]
			nowval = vals_sorted[j]
			while (nowleft <= len) && (mm_sorted[nowleft] < nowcut+1.0)
				nowleftsum[vals_sorted[nowleft]] += b_sorted[nowleft]
				countleft[vals_sorted[nowleft]] +=1
				nowleft+=1
			end
			while (nowright <= len) && (mm_sorted[nowright] <= nowcut-1.0)
				nowrightsum[vals_sorted[nowright]] -= b_sorted[nowright]
				countright[vals_sorted[nowright]] -=1
				nowright+=1
			end

			c_p = 0.0
			for k=1:nowval-1
				c_p += (countright[k]*(b_sorted[j]-1) - nowrightsum[k])
#				c_p += (countright[k]*b_sorted[j] - nowrightsum[k])
			end
			for k=nowval+1:nlevels
				c_p += (countleft[k]*(b_sorted[j]+1) - nowleftsum[k])
#				c_p += (countleft[k]*b_sorted[j] - nowleftsum[k])
			end
#			c_p *= 2.0
			p = d2bar_sorted[j];

			g1[:,p] += ui * (c_p*2.0)
		end

		else
		t = zeros(len);
		for j in 1:(len - 1)
#			J = d2_bar[j];
			for k in (j + 1):len
#				K = d2_bar[k];
				if vals_d2_bar[j] > vals_d2_bar[k]
				totalnum += 1
#		println(ff, i, " ", J, " ", K);
					mask = mm[j] - mm[k];
					if mask < 1.0
						s_jk = 2.0 * (mask - 1.0)
						t[j] += s_jk
						t[k] -= s_jk
					end
				elseif vals_d2_bar[k] > vals_d2_bar[j]
				totalnum +=1
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
			g1[:,J] += ui * t[j]
		end
	end
	end

#println(norm(g-g1)/norm(g))
	return g1
end


function comp_m(U, V, X, d1, d2, rows, vals, cols)

	mvals = zeros(nnz(X))
	cc=0
	for i=1:d1
		tmp = nzrange(X,i)
		d2_bar = rows[tmp];
		ui = U[:,i]
		for j in d2_bar
			cc+=1
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

function compute_Ha_newnewnew(a, m, U, X, r, d1, d2, lambda, rows, vals, infor_list)
	Ha = lambda * a
#	Ha1 = lambda*a

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

#	println(len);


#	if len >= 10
	if len >= 50
#		levels = sort(unique(vals_d2_bar))
#		nlevels = size(levels)[1]
#		perm_ind = sortperm(mm)
#		mm_sorted = mm[perm_ind]
#		vals_sorted = vals_d2_bar[perm_ind]
#		for j=1:len
#			for k=1:nlevels
#				if vals_sorted[j] == levels[k]
#					vals_sorted[j] = k
#					break
#				end
#			end
#		end
#		d2bar_sorted = d2_bar[perm_ind]
#		countright = zeros(Int, nlevels)
#		for j=1:len
#			countright[vals_sorted[j]] +=1
#		end


		nlevels = infor_list[i].nlevels
#		println(countright, " ", infor_list[i].countright)
		countright = copy(infor_list[i].countright)
		perm_ind = infor_list[i].perm_ind
		vals_sorted = infor_list[i].vals_sorted
		mm_sorted = infor_list[i].mm_sorted
		d2bar_sorted = infor_list[i].d2bar_sorted

		nowleft = 1
		nowright = 1
		countleft = zeros(Int, nlevels)

		b_sorted = b[perm_ind]
		allsum = zeros(nlevels)
		for j=1:len
			allsum[vals_sorted[j]] += b_sorted[j]
		end

		nowleftsum = zeros(nlevels)
		nowrightsum = allsum
		for j = 1:len
			nowcut = mm_sorted[j]
			nowval = vals_sorted[j]
			while (nowleft <= len) && (infor_list[i].mm_sorted[nowleft] < nowcut+1.0)
				nowleftsum[vals_sorted[nowleft]] += b_sorted[nowleft]
				countleft[vals_sorted[nowleft]] +=1
				nowleft+=1
			end
			while (nowright <= len) && (infor_list[i].mm_sorted[nowright] <= nowcut-1.0)
				nowrightsum[vals_sorted[nowright]] -= b_sorted[nowright]
				countright[vals_sorted[nowright]] -=1
				nowright+=1
			end

			c_p = 0.0
			for k=1:nowval-1
				c_p += (countright[k]*b_sorted[j] - nowrightsum[k])
			end
			for k=nowval+1:nlevels
				c_p += (countleft[k]*b_sorted[j] - nowleftsum[k])
			end
			p = d2bar_sorted[j];
			Ha[(p - 1) * r + 1 : p * r] += 2.0*c_p*ui
		end

	else
		for j in 1:len
			p = d2_bar[j];
			c_p = 0.0
			for k in 1:len
				if vals_d2_bar[j] == vals_d2_bar[k]
					continue
				elseif vals_d2_bar[j] > vals_d2_bar[k]
					y_ipq = 1.0
				elseif vals_d2_bar[k] > vals_d2_bar[j]
					y_ipq = -1.0
				end
				mask = y_ipq * (mm[j] - mm[k])
				if mask < 1.0
					c_p += 2.0 * (b[j] - b[k])
				end
			end
			Ha[(p - 1) * r + 1 : p * r] += c_p*ui
		end
	end
	end

	return Ha
end	



function compute_Ha_newnew(a, m, U, X, r, d1, d2, lambda, rows, vals)
	Ha = lambda * a
#	Ha1 = lambda*a

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
	if len >= 0

		levels = sort(unique(vals_d2_bar))
		nlevels = size(levels)[1]
		perm_ind = sortperm(mm)
		mm_sorted = mm[perm_ind]
		b_sorted = b[perm_ind]
		vals_sorted = vals_d2_bar[perm_ind]
		for j=1:len
			for k=1:nlevels
				if vals_sorted[j] == levels[k]
					vals_sorted[j] = k
					break
				end
			end
		end
		d2bar_sorted = d2_bar[perm_ind]
		allsum = zeros(nlevels)
		for j=1:len
			allsum[vals_sorted[j]] += b_sorted[j]
		end


		nowleft = 1
		nowright = 1
		countleft = zeros(Int, nlevels)
		countright = zeros(Int, nlevels)
		for j=1:len
			countright[vals_sorted[j]] +=1
		end

		nowleftsum = zeros(nlevels)
		nowrightsum = allsum
		cplist = zeros(len)
		for j = 1:len
			nowcut = mm_sorted[j]
			nowval = vals_sorted[j]
			while (nowleft <= len) && (mm_sorted[nowleft] < nowcut+1.0)
				nowleftsum[vals_sorted[nowleft]] += b_sorted[nowleft]
				countleft[vals_sorted[nowleft]] +=1
				nowleft+=1
			end
			while (nowright <= len) && (mm_sorted[nowright] <= nowcut-1.0)
				nowrightsum[vals_sorted[nowright]] -= b_sorted[nowright]
				countright[vals_sorted[nowright]] -=1
				nowright+=1
			end

			c_p = 0.0
			for k=1:nowval-1
				c_p += (countright[k]*b_sorted[j] - nowrightsum[k])
			end
			for k=nowval+1:nlevels
				c_p += (countleft[k]*b_sorted[j] - nowleftsum[k])
			end
			c_p *= 2.0
			p = d2bar_sorted[j];

			cc=0
			for kk=((p-1)*r+1):(p*r)
				cc+=1
				Ha[kk] += c_p*ui[cc]
			end
#			Ha[(p - 1) * r + 1 : p * r] += 2.0*c_p*ui
		end
	else
		for j in 1:len
			p = d2_bar[j];
			c_p = 0.0
			for k in 1:len
				if vals_d2_bar[j] == vals_d2_bar[k]
					continue
				elseif vals_d2_bar[j] > vals_d2_bar[k]
					y_ipq = 1.0
				elseif vals_d2_bar[k] > vals_d2_bar[j]
					y_ipq = -1.0
				end
				mask = y_ipq * (mm[j] - mm[k])
				if mask < 1.0
					c_p += 2.0 * (b[j] - b[k])
				end
			end
			Ha[(p - 1) * r + 1 : p * r] += c_p*ui
		end
	end
	end

	return Ha
end	



function compute_Ha_new(a, m, U, X, r, d1, d2, lambda, rows, vals)
	Ha = lambda * a
#	Ha1 = lambda*a
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
			cc=0
			for kk=((p-1)*r+1):(p*r)
				cc+=1
				Ha[kk] += cpvals[j]*ui[cc]
			end
#			Ha[(p - 1) * r + 1 : p * r] += cpvals[j]*ui
		end

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

function solve_delta(g, m, U, X, r, d1, d2, lambda, rows, vals, infor_list)
	# use linear conjugate grad descent
	delta = zeros(size(g)[1])
	rr = -g
	p = -rr
	err = norm(rr) * 10.0^-2
	for k in 1:10
		#Hp = compute_Ha(p, m, U, X, r, d1, d2, lambda, rows, vals)
#@time Hp = compute_Ha_new(p, m, U, X, r, d1, d2, lambda, rows, vals)
		Hp = compute_Ha_newnew(p, m, U, X, r, d1, d2, lambda, rows, vals)
#		Hp = compute_Ha_newnewnew(p, m, U, X, r, d1, d2, lambda, rows, vals, infor_list)
#		println("test Hp, Hp1, diff = ", vecnorm(Hp-Hp1)/vecnorm(Hp))
		alpha = -dot(rr, p) / dot(p, Hp)
		delta += alpha * p
		rr += alpha * Hp
#		println("rr: ", vecnorm(rr))
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
	res1 = 0.0
	res1 = lambda / 2.0 * (vecnorm(U) ^ 2 +vecnorm(V) ^ 2)
	for i in 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		vals_d2_bar = vals[tmp];
		len = size(d2_bar)[1];
		mm = nonzeros(m[:,i])

		levels = sort(unique(vals_d2_bar))
		nlevels = size(levels)[1]
		perm_ind = sortperm(mm)
		mm_sorted = mm[perm_ind]
#		b_sorted = b[perm_ind]
		vals_sorted = vals_d2_bar[perm_ind]
		for j=1:len
			for k=1:nlevels
				if vals_sorted[j] == levels[k]
					vals_sorted[j] = k
					break
				end
			end
		end
#		d2bar_sorted = d2_bar[perm_ind]
#		allsum = zeros(nlevels)
#		for j=1:len
#			allsum[vals_sorted[j]] += b_sorted[j]
#		end


		nowleft = 1
#		nowright = 1
		countleft = zeros(Int, nlevels)
#		countright = zeros(Int, nlevels)
#		for j=1:len
#			countright[vals_sorted[j]] +=1
#		end

		nowleftsum = zeros(nlevels)
		nowleftsqsum = zeros(nlevels)
#		nowrightsum = allsum
#		cplist = zeros(len)
		for j = 1:len
			nowcut = mm_sorted[j]
			nowval = vals_sorted[j]
			while (nowleft <= len) && (mm_sorted[nowleft] < nowcut+1.0)
				nowleftsum[vals_sorted[nowleft]] += (mm_sorted[nowleft]-1)
				nowleftsqsum[vals_sorted[nowleft]] += ((mm_sorted[nowleft]-1)^2)
				countleft[vals_sorted[nowleft]] +=1
				nowleft+=1
			end
#			while (nowright <= len) && (mm_sorted[nowright] <= nowcut-1.0)
#				nowrightsum[vals_sorted[nowright]] -= b_sorted[nowright]
#				countright[vals_sorted[nowright]] -=1
#				nowright+=1
#			end

#			c_p = 0.0
#			for k=1:nowval-1
#				c_p += (countright[k]*b_sorted[j] - nowrightsum[k])
#			end
			for k=nowval+1:nlevels
				res1 += (countleft[k]*(nowcut^2)-2.0*nowcut*nowleftsum[k] + nowleftsqsum[k])
			end
		end
	end

#	res = 0.0
#	res = lambda / 2.0 * (vecnorm(U) ^ 2 +vecnorm(V) ^ 2)
#	for i in 1:d1
#		tmp = nzrange(X, i)
#		d2_bar = rows[tmp];
#		vals_d2_bar = vals[tmp];
#		len = size(d2_bar)[1];
#		mm = nonzeros(m[:,i])
#
#		for j in 1:(len - 1)
#			for k in (j + 1):len
#				if vals_d2_bar[j] == vals_d2_bar[k]
#					continue
#				elseif vals_d2_bar[j] > vals_d2_bar[k]
#					y_ipq = 1.0
#				elseif vals_d2_bar[k] > vals_d2_bar[j]
#					y_ipq = -1.0
#				end
#				mask = y_ipq * (mm[j]-mm[k])
#				if mask < 1.0
#					res += (1.0 - mask) ^ 2
#				end
#			end
#		end
#	end
	return res1
end

function precompute_ui(V, X, r, d2, lambda, rows, vals, mm, i)

	tmp = nzrange(X, i)
	d2_bar = rows[tmp]
	vals_d2_bar = vals[tmp]
	len = size(d2_bar)[1]

#mm = nonzeros(m[:,i])

	levels = sort(unique(vals_d2_bar))
	nlevels = size(levels)[1]
	perm_ind = sortperm(mm)
	mm_sorted = mm[perm_ind]
	vals_sorted = vals_d2_bar[perm_ind]
	for j=1:len
		for k=1:nlevels
			if vals_sorted[j] == levels[k]
				vals_sorted[j] = k
				break
			end
		end
	end
	d2bar_sorted = d2_bar[perm_ind]

	nowleft = 1
	nowright = 1
	countright = zeros(Int, nlevels)
	for j=1:len
		countright[vals_sorted[j]] +=1
	end

	infor_ui = infor(levels, nlevels, perm_ind, mm_sorted, vals_sorted, d2bar_sorted, countright)
	return infor_ui
end



function precompute(U, V, X, r, d1, d2, lambda, rows, vals, m)
	infor_list = Array{infor}(d1)

	for i in 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp]
		vals_d2_bar = vals[tmp]
		len = size(d2_bar)[1]

		mm = nonzeros(m[:,i])

		levels = sort(unique(vals_d2_bar))
		nlevels = size(levels)[1]
		perm_ind = sortperm(mm)
		mm_sorted = mm[perm_ind]
		vals_sorted = vals_d2_bar[perm_ind]
		for j=1:len
			for k=1:nlevels
				if vals_sorted[j] == levels[k]
					vals_sorted[j] = k
					break
				end
			end
		end
		d2bar_sorted = d2_bar[perm_ind]

		nowleft = 1
		nowright = 1
		countright = zeros(Int, nlevels)
		for j=1:len
			countright[vals_sorted[j]] +=1
		end

		infor_list[i] = infor(levels, nlevels, perm_ind, mm_sorted, vals_sorted, d2bar_sorted, countright)
	end
	return infor_list 
end

function update_V(U, V, X, r, d1, d2, lambda, rows, vals, stepsize, cols)
	# g,m = obtain_g(U, V, X, d1, d2, lambda, rows, vals)
	m = comp_m(U, V, X, d1, d2, rows, vals, cols);
	infor_list =  precompute(U, V, X, r, d1, d2, lambda, rows, vals, m);
	infor_list = [];
  	g = obtain_g_new(U, V, X, d1, d2, lambda, rows, vals,m)
	delta = solve_delta(vec(g), m, U, X, r, d1, d2, lambda, rows, vals, infor_list)
	delta = reshape(delta, size(V))
	prev_obj = objective(m, U, V, X, d1, lambda, rows, vals)

	Vold = V;
	s = float(stepsize)
	new_obj=0.0
	for iter=1:20
		V = Vold - s * delta
		m = comp_m(U, V, X, d1, d2, rows, vals, cols);
		new_obj = objective(m, U, V, X, d1, lambda, rows, vals)
		#println("Line Search iter ", iter, " Prev Obj ", prev_obj, " New Obj ", new_obj)
		if (new_obj < prev_obj)
			break
		else
			s/=2.0
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

function solve_delta_u(g, D, lambda, i, V, r, d2, vals, X, rows, mm, infor_ui)
	# use linear conjugate grad descent
	delta = zeros(size(g)[1])
	rr = -g
	p = -rr
	err = norm(rr) * 10.0^-2
	for k in 1:10
#Hp1 = obtain_Hs(p, A, D, V_bar, lambda)
#	Hp = obtain_Hs_new(i, V, X, r, d2, rows, vals, lambda, D, p);
	Hp = obtain_Hs_newnew(i, V, X, r, d2, rows, vals, lambda, D, p, mm, infor_ui);
#	println("diff: ", norm(Hp-Hp1))
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
	res1 = lambda / 2 * (vecnorm(ui) ^ 2)
	tmp = nzrange(X, i)
	d2_bar = rows[tmp];
	vals_d2_bar = vals[tmp];
	len = size(d2_bar)[1];

	levels = sort(unique(vals_d2_bar))
	nlevels = size(levels)[1]
	perm_ind = sortperm(mm)
	mm_sorted = mm[perm_ind]
	vals_sorted = vals_d2_bar[perm_ind]
	for j=1:len
		for k=1:nlevels
			if vals_sorted[j] == levels[k]
				vals_sorted[j] = k
				break
			end
		end
	end
#	d2bar_sorted = d2_bar[perm_ind]
#	allsum = zeros(nlevels)
#	for j=1:len
#		allsum[vals_sorted[j]] += mm_sorted[j]
#	end

	nowleft = 1
#	nowright = 1
	countleft = zeros(Int, nlevels)
#	countright = zeros(Int, nlevels)
#	for j=1:len
#		countright[vals_sorted[j]] +=1
#	end

	nowleftsum = zeros(nlevels)
	nowleftsqsum = zeros(nlevels)
#	nowrightsum = allsum
	for j = 1:len
		nowcut = mm_sorted[j]
		nowval = vals_sorted[j]
		while (nowleft <= len) && (mm_sorted[nowleft] < nowcut+1.0)
			nowleftsum[vals_sorted[nowleft]] += (mm_sorted[nowleft]-1)
			nowleftsqsum[vals_sorted[nowleft]] += ((mm_sorted[nowleft]-1)^2)
			countleft[vals_sorted[nowleft]] +=1
			nowleft+=1
		end
#		while (nowright <= len) && (mm_sorted[nowright] <= nowcut-1.0)
#			nowrightsum[vals_sorted[nowright]] -= b_sorted[nowright]
#			countright[vals_sorted[nowright]] -=1
#			nowright+=1
#		end

#		c_p = 0.0
#		for k=1:nowval-1
#			c_p += (countright[k]*b_sorted[j] - nowrightsum[k])
#		end
		for k=nowval+1:nlevels
			res1 += ( countleft[k]*(nowcut^2) - 2.0*nowcut*nowleftsum[k] + nowleftsqsum[k])
#			c_p += (countleft[k]*b_sorted[j] - nowleftsum[k])
		end
#		p = d2bar_sorted[j];
#		g += c_p*2.0*V[:,p];
	end

#	res = lambda / 2 * (vecnorm(ui) ^ 2)
#	tmp = nzrange(X, i)
#	d2_bar = rows[tmp];
#	vals_d2_bar = vals[tmp];
#	len = size(d2_bar)[1];
#	for j in 1:(len - 1)
#		for k in (j + 1):len
#			if vals_d2_bar[j] == vals_d2_bar[k]
#				continue
#			elseif vals_d2_bar[j] > vals_d2_bar[k]
#				y_ipq = 1.0
#			else
#				y_ipq = -1.0
#			end
#			mask = y_ipq * (mm[j] - mm[k])
#			if mask < 1.0
#				res += (1.0 - mask) ^ 2
#			end
#		end
#	end
	
	return res1
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

function obtain_g_u_newnew(i, ui, V, X, r, d2, rows, vals, lambda, mm, infor_ui)
	
	g1 = lambda * ui
	tmp = nzrange(X, i)
	len = size(tmp)[1];
	if len==0
		return zeros(r), zeros(r), 0
	end
	d2_bar = rows[tmp];

	vals_d2_bar = vals[tmp];
	num = round(Int64, len*(len-1)/2)
	D = zeros(Int, num)

	nlevels = infor_ui.nlevels
	countright = copy(infor_ui.countright)
	perm_ind = infor_ui.perm_ind
	vals_sorted = infor_ui.vals_sorted
	mm_sorted = infor_ui.mm_sorted
	d2bar_sorted = infor_ui.d2bar_sorted

	nowleft = 1
	nowright = 1
	countleft = zeros(Int, nlevels)

	b_sorted = mm[perm_ind]
	allsum = zeros(nlevels)
	for j=1:len
		allsum[vals_sorted[j]] += b_sorted[j]
	end


	nowleftsum = zeros(nlevels)
	nowrightsum = allsum
	for j = 1:len
		nowcut = mm_sorted[j]
		nowval = vals_sorted[j]
		while (nowleft <= len) && (mm_sorted[nowleft] < nowcut+1.0)
			nowleftsum[vals_sorted[nowleft]] += b_sorted[nowleft]
			countleft[vals_sorted[nowleft]] +=1
			nowleft+=1
		end
		while (nowright <= len) && (mm_sorted[nowright] <= nowcut-1.0)
			nowrightsum[vals_sorted[nowright]] -= b_sorted[nowright]
			countright[vals_sorted[nowright]] -=1
			nowright+=1
		end

		c_p = 0.0
		for k=1:nowval-1
			c_p += (countright[k]*(b_sorted[j]-1) - nowrightsum[k])
		end
		for k=nowval+1:nlevels
			c_p += (countleft[k]*(b_sorted[j]+1) - nowleftsum[k])
		end
		p = d2bar_sorted[j];
		g1 += c_p*2.0*V[:,p];
	end

#	g = zeros(r)
#	tmp_vals = zeros(len)
#
#	c = 0
#	for j = 1:len-1
#		for k = (j + 1):len
#			if vals_d2_bar[j] == vals_d2_bar[k]
#				continue
#			elseif vals_d2_bar[j] > vals_d2_bar[k]
#				y_ipq = 1.0
#			else
#				y_ipq = -1.0
#			end
#			c+=1
#				mask = y_ipq * (mm[j]-mm[k])
#			if mask < 1.0
#						D[c] = 1.0;
#						aaa = 2*(1-mask)*y_ipq;
#						tmp_vals[j] -= aaa;
#						tmp_vals[k] += aaa;
#			end
#		end
#	end
#
#	for j in 1:len
#		p = d2_bar[j];
#		g += tmp_vals[j]*V[:,p];
#	end
#
#	g += lambda * ui
	c=1
	D=0
	return g1, D, c
end


function obtain_g_u_new(i, ui, V, X, r, d2, rows, vals, lambda, mm)
	tmp = nzrange(X, i)
	len = size(tmp)[1];
	if len==0
		return zeros(r), zeros(r), 0
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


function obtain_Hs_newnew(i, V, X, r, d2, rows, vals, lambda, D, s, mm, infor_ui)
	tmp = nzrange(X, i)
	d2_bar = rows[tmp];
	vals_d2_bar = vals[tmp];
	len = size(d2_bar)[1];

	m = zeros(len)
	c=0;
	for j in d2_bar
		c+=1;
		m[c] = dot(s,V[:,j])
	end
	g = zeros(r)


#	levels = sort(unique(vals_d2_bar))
#	nlevels = size(levels)[1]
#	perm_ind = sortperm(mm)
#	mm_sorted = mm[perm_ind]
#	b_sorted = m[perm_ind]
#	vals_sorted = vals_d2_bar[perm_ind]
#	for j=1:len
#		for k=1:nlevels
#			if vals_sorted[j] == levels[k]
#				vals_sorted[j] = k
#				break
#			end
#		end
#	end
#	d2bar_sorted = d2_bar[perm_ind]
#	allsum = zeros(nlevels)
#	for j=1:len
#		allsum[vals_sorted[j]] += b_sorted[j]
#	end
#
#	nowleft = 1
#	nowright = 1
#	countleft = zeros(Int, nlevels)
#	countright = zeros(Int, nlevels)
#	for j=1:len
#		countright[vals_sorted[j]] +=1
#	end


	nlevels = infor_ui.nlevels
	countright = copy(infor_ui.countright)
	perm_ind = infor_ui.perm_ind
	vals_sorted = infor_ui.vals_sorted
	mm_sorted = infor_ui.mm_sorted
	d2bar_sorted = infor_ui.d2bar_sorted

	nowleft = 1
	nowright = 1
	countleft = zeros(Int, nlevels)

	b_sorted = m[perm_ind]
	allsum = zeros(nlevels)
	for j=1:len
		allsum[vals_sorted[j]] += b_sorted[j]
	end


	nowleftsum = zeros(nlevels)
	nowrightsum = allsum
	for j = 1:len
		nowcut = mm_sorted[j]
		nowval = vals_sorted[j]
		while (nowleft <= len) && (mm_sorted[nowleft] < nowcut+1.0)
			nowleftsum[vals_sorted[nowleft]] += b_sorted[nowleft]
			countleft[vals_sorted[nowleft]] +=1
			nowleft+=1
		end
		while (nowright <= len) && (mm_sorted[nowright] <= nowcut-1.0)
			nowrightsum[vals_sorted[nowright]] -= b_sorted[nowright]
			countright[vals_sorted[nowright]] -=1
			nowright+=1
		end

		c_p = 0.0
		for k=1:nowval-1
			c_p += (countright[k]*b_sorted[j] - nowrightsum[k])
		end
		for k=nowval+1:nlevels
			c_p += (countleft[k]*b_sorted[j] - nowleftsum[k])
		end
		p = d2bar_sorted[j];
		g += c_p*2.0*V[:,p];
	end

#	tmp_vals = zeros(len)
#	c = 0
#	for j in 1:len
#		for k in (j + 1):len
#			if vals_d2_bar[j] == vals_d2_bar[k]
#				continue
#			end
#			c+=1
#			mask = m[j] - m[k];
#			if D[c] > 0.0
#						aaa = 2.0*mask;
#						tmp_vals[j] += aaa;
#						tmp_vals[k] -= aaa;
#			end
#		end
#	end
#
#	for j in 1:len
#		p = d2_bar[j];
#		g += tmp_vals[j]*V[:,p];
#	end

	g += lambda * s
	return g
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
	
	infor_ui = precompute_ui(V, X, r, d2, lambda, rows, vals, mm, i);
#	g, D,c = obtain_g_u_new(i, ui, V, X, r, d2, rows, vals, lambda, mm);
	g, D,c = obtain_g_u_newnew(i, ui, V, X, r, d2, rows, vals, lambda, mm, infor_ui);

	prev_obj = objective_u_new(i, X, lambda, rows, vals, ui, mm);
	if (c == 0.0) || (norm(g)<1e-4)
		return ui, prev_obj, mm
	end
	delta = solve_delta_u(g, D, lambda, i, V, r, d2, vals, X, rows, mm, infor_ui)

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
		prev = 0
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
#println("prev value: ", prev)
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
	println("Training dataset ", train, " and test dataset ", test, " are loaded. \n There are ", n, " users and ", msize, " items in the dataset.")

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
