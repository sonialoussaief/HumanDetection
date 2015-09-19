function p = createP(X1,X2)
% function p = createP(X1,X2):
% create the two columns of P
% input  X1  world coordinates of cube corner
%        X2  image coordinates of cube corner
% result  p  two columns of P
%     P = [cube_point,1];
    p = zeros(2,12);
    p(1, 1:4) = X1';
    p(1, 9:12) = -X2(1)*X1';
    
    p(2, 5:8) = X1';
    p(2, 9:12) = -X2(2)*X1';