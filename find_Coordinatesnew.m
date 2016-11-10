function C = find_Coordinatesnew(image)
%image is already processed such that the number plate is highlighted.

[r,c] = size(image);
C = zeros(2,2);

H = zeros(1,c);
V = zeros(1,r);
for i=1:c
    for j=1:r
        H(1,i) = H(1,i) + image(j,i);
    end
end

for i=1:r
    for j=1:c
        V(1,i) = V(1,i) + image(i,j);
    end
end

HS = smooth(H,50);
VS = smooth(V,50);
Hs = HS;            %later change HS and Hs accordingly as per 
Vs = VS;            %the general position of car.

dim = [r,c];

h = 1:dim(1,2);
v = 1:dim(1,1);

figure(50), plot(h,HS);
figure(51), plot(v,VS);
[Mx,Cx] = max(HS);
[My,Cy] = max(VS);

i = 1;
k = Cx;
while(i>0)
    if k==1
        break;
    end
    i = Hs(k,1)-(0.1)*Mx;           %additional term to truncate at 90%of max value
    k=k-1;
end
C(1,1) = k+1;

i = 1;
k = Cx;
while(i>0)
    if k == dim(1,2)
        break;
    end
    i = Hs(k,1)-(0.1)*Mx;
    k=k+1;
end
C(1,2) = k-1;

i = 1;
k = Cy;
while(i>0)
    if k==1
        break;
    end
    i = Vs(k,1)-(0.1)*My;
    k=k-1;
end
C(2,1) = k+1;

i = 1;
k = Cy;
while(i>0)
    if k == dim(1,1)
        break;
    end
    i = Vs(k,1)-(0.1)*My;
    k=k+1;
end
C(2,2) = k-1;
