function f = fun(idx,kneighbor_mat,data,x)
k=25;
a = data(idx,:);
b = zeros(1,16);
for i = 1:k
    n_id = kneighbor_mat(idx,i);
    c = data(n_id,:);
    b = b + c*x(i);
end

f = norm(a-b);

end




