
%INPUT
% mu_s              (3,N)
% mu_tex            (N*3,1)
% sigma             (199,1)
% shape_eigenvec    (N*3,199)
%
%OUTPUT
% xw                (3,N)
% alphas            (199,1)
function [xw,alphas] = generateRandomFace(mu_s,sigma,shape_eigenvec)

    s = size(mu_s,1);
    N = s/3;
    alphas = randn(199,1);
    delta = shape_eigenvec*diag(sigma)*alphas;
%     delta = reshape(delta,[3,size(mu_s,2)]);
    xw = mu_s + delta;
    xw = reshape(xw,3,N);
end