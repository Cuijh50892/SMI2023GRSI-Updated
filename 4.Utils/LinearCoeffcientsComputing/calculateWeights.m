data = csvread('codesY.csv');
data_num = size(data,1);
dist_mat = zeros(data_num,data_num);
k=25;
kneighbor_mat = zeros(data_num,k);
kweights_mat = zeros(data_num,k);
check_list = zeros(1,data_num); 

for idx = 1:data_num
    a = data(idx,:);
    idlist = randperm(data_num);
    idlist = sort(idlist); 
    for jdx = 1:data_num
        if(idx==jdx)
            continue;
        end
        b = data(jdx,:);
        dist = norm(a-b);
        dist_mat(idx,jdx)=dist;      
    end
    distlist = dist_mat(idx,:);
    [sortdist,index]=sort(distlist);
    idlist=idlist(index);
    kneighbor_mat(idx,:) = idlist(2:k+1);
end

kneighbor_mat_out = kneighbor_mat -1; 

x0 = zeros(1,k);
for idx = 1:k
    x0(idx) = 0.1;
end

parfor idx = 1:data_num
    %y = fun(idx,kneighbor_mat,data,x0);
    [x,y]=fmincon(@(x)fun(idx,kneighbor_mat,data,x),x0,[],[],[],[],[],[],'ceq');
    disp(idx);
    kweights_mat(idx,:) = x;
    check_list(idx) = y;
end

csvwrite('neighborsY.csv',kneighbor_mat_out, 0, 0)
csvwrite('weightsY.csv',kweights_mat, 0, 0)



