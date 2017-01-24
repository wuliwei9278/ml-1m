#split_train_test("MovieLens1m.csv", 0.9) 
#convertRatingBinary("MovieLens1m_train")
#convertRatingBinary("MovieLens1m_test")

# file name (assume text file seprated by comma and starting with 0, set sampling rate for training data set
function split_train_test(f, rate) 
	X = readdlm(f, ',' , Int64);
	x = vec(X[:,1]);
	y = vec(X[:,2]);
	v = vec(X[:,3]);
	g = rand(length(v))
	v_train = v[g.<=rate]; x_train = x[g.<=rate]; y_train = y[g.<=rate];
	f = split(f,'.')[1];
	ff = open(string(f,"_train"), "w");
	for i = 1:length(v_train)
		println(ff, x_train[i], ",", y_train[i], ",", v_train[i]);
	end
	close(ff)
	ff = open(string(f,"_test"), "w");
	v_test = v[g.>rate]; x_test = x[g.>rate]; y_test = y[g.>rate];
	for i = 1:length(v_test)
		println(ff, x_test[i] + 1, " ", y_test[i] + 1);
	end
	close(ff)
end 


function convertRatingBinary(f)
	X = readdlm(f, ',' , Int64);
	x = vec(X[:,1]) + 1; # userid starting from 0
	y = vec(X[:,2]) + 1; # same for movieid
	v = vec(X[:,3]);
	n = 6040; msize = 3952;
	X = sparse(x, y, v, n, msize); # userid by movieid
	# julia column major 
	# now moveid by userid
	X = X'; 
	rows = rowvals(X);
	vals = nonzeros(X);
	d2, d1 = size(X);
	ff = open(string(f,"_Bin"), "w");
	for i = 1:d1
		tmp = nzrange(X, i)
		d2_bar = rows[tmp];
		vals_d2_bar = vals[tmp];
		len = size(d2_bar)[1];
		for j in 1:(len - 1)
			J = d2_bar[j];
			for k in (j + 1):len
				K = d2_bar[k];
				if vals_d2_bar[j] > vals_d2_bar[k]
					println(ff, i, " ", J, " ", K);
				elseif vals_d2_bar[k] > vals_d2_bar[j]
					println(ff, i, " ", K, " ", J);
				end
			end
		end
	end
	close(ff)
end