c = 0
for i = 1:d1
	tmp = nzrange(X, i)
	len = length(tmp)
	vals_d2_bar = vals[tmp]
	tmp = tmp[vals_d2_bar .== 1]
	tr = length(tmp)
	c = c + tr * (len - tr)
end
println(c / d1)
