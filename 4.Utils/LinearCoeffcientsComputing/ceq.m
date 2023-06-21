function [g,h] = ceq(x)
g = 0;
h = -1;
k=25;
for i = 1:k %k
    h = h + x(i);
end
end