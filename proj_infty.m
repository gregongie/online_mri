function y = proj_infty(x,lam)
    if nargin < 2
        lam = 1; %default: projection onto unit \ell^\infty ball
    end
    y = x./max(abs(x)/lam,1);
end
