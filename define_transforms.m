function [ T, Tt ] = define_transforms( settings )
dims = settings.dims;
b = settings.b;
alpha = settings.alpha;

% Forward finite difference operator 
% (with circular boundary conditions)
function DU = D(U)
    U = reshape(U,dims);
    DU(:,:,1) = [diff(U,1,2), U(:,1) - U(:,end)];
    DU(:,:,2) = [diff(U,1,1); U(1,:) - U(end,:)];
    DU = DU(:);
end

% Divergence operator (transpose of gradient)
function DtV = Dt(V)
    V = reshape(V,[dims,2]);
    X1 = V(:,:,1);
    X2 = V(:,:,2);
    DtV = [X1(:,end) - X1(:,1), -diff(X1,1,2)];
    DtV = DtV + [X2(end,:) - X2(1,:); -diff(X2,1,1)];
    DtV = DtV(:);
end

function DU = Dtemp(U)
    U = reshape(U,[prod(dims),b]);
    DU = diff(U,1,2);
    DU = DU(:);
end

function DtV = Dttemp(V)
    V = reshape(V,[prod(dims),b-1]);
    V = [zeros(prod(dims),1),V,zeros(prod(dims),1)];
    DtV = -diff(V,1,2);
    DtV = DtV(:);
end

function Tout = Tdef(U)
    U = reshape(U,[prod(dims),b]);
    Tout = [];
    for i=1:b
        Tout = [Tout; D(U(:,i))];
    end
    Tout = [Tout; alpha*Dtemp(U)];
end

function Ttout = Ttdef(V)
    V1 = V(1:(b*2*prod(dims)));
    V2 = V((b*2*prod(dims)+1):end);
    V1 = reshape(V1,[2*prod(dims),b]);
    Ttout = zeros([prod(dims),b]);
    for i=1:b
        Ttout(:,i) = Dt(V1(:,i));
    end
    Ttout = Ttout(:) + alpha*Dttemp(V2);
    Ttout = reshape(Ttout,[prod(dims),b]);
end

T = @(x) Tdef(x);
Tt = @(x) Ttdef(x);

end

