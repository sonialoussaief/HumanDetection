% ---------- 3d camera calibration
% plot world coordinates of the cube vertices
im = imread('cali.jpg');
figure;
imshow(im);
hold on;
[x,y] = ginput(8);
X2(1,:) = x';
X2(2,:) = y';
X2(3,:) = ones(1,8);
plot(X2(1,:),X2(2,:),'gx');
hold on;

X1 = [0 1.8 1.8 0 0 1.8 1.8 0;    % world coordinates
      0 0 1.2 1.2 0 0 1.2 1.2;
      0 0 0 0 0.9 0.9 0.9 0.9;
      1 1 1 1 1 1 1 1]*1000;
for i = 1 : 8
    p = createP(X1(:,i),X2(:,i));
    P(2*i-1:2*i,:) = p;
end
P

% M
[~,~,V] = svd(P);
M(1,:) = V(1:4,12)';
M(2,:) = V(5:8,12)';
M(3,:) = V(9:12,12)';
M

% compute camera center
[~,~,V] = svd(M);
cc = V(1:3,4)/V(4,4);
cc

% compute first 3 columns of M
m = M(1:3,1:3)/M(3,3);
m
    
% compute Rx, Thetax, N 
cos = m(3,3) / sqrt(m(3,3)*m(3,3) + m(3,2)*m(3,2));
sin = -m(3,2) / sqrt(m(3,3)*m(3,3) + m(3,2)*m(3,2));
Rx = [1 0 0; 0 cos -sin; 0 sin cos]
N = m*Rx
thetax = asin(sin) / pi*180

% compute Rz, Thetaz
cos = N(2,2) / sqrt(N(2,1)*N(2,1) + N(2,2)*N(2,2));
sin = -N(2,1) / sqrt(N(2,1)*N(2,1) + N(2,2)*N(2,2));
Rz = [cos -sin 0; sin cos 0; 0 0 1]
thetaz = asin(sin) / pi*180

% compute K
K = N * Rz;
K = K / K(3,3);
% test
X_1 = [0 1 1 0 0 1 1 0;    % world coordinates
      0 0 1.2 1.2 0 0 1.2 1.2;
      0 0 0 0 1.5 1.5 1.5 1.5;
      1 1 1 1 1 1 1 1]*1000;
X_2 = M * X_1;
X_2 = X_2./[X_2(3,:);X_2(3,:);X_2(3,:)];
c = uint16(X_2(1,:));
r = uint16(X_2(2,:));
plot(c,r,'rx'); 

save('M.mat', M);