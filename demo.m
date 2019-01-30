%% RUFFed GROUSE MRI demo
% Online MRI reconstruction with low-rank plus
% sparse model using the RUFFed GROUSE algorithm from:
%
% Ongie, G., Dewangan, S., Fessier, J. A., & Balzano, L. (2017). 
% Online dynamic MRI reconstruction via robust subspace tracking. 
% Proceedings of IEEE GlobalSIP (pp. 1180-1184).
% 
% Requires Michigan IRT toolbox to be installed on the path:
% https://web.eecs.umich.edu/~fessler/code/
%
% Greg Ongie, Jan. 29 2019
% e-mail: gongie@uchicago.edu
%%
clear; clc; close all;
addpath(genpath(pwd));
%% load dataset
load data/mrxcat.mat;
load data/mrxcat_R8.mat;
%% settings and parameters
[nx,ny,nf,nc] = size(kdata);
settings.dims = [nx,ny]; %frame dimensions
settings.nf = nf; %number of frames
settings.nc = nc; %number of coils;
settings.b = 6; %batch size
settings.r = 1; %rank
settings.pd.niters = 20;
settings.pd.tau = 1;
settings.pd.sigma = 0.1;
%% define batches
b = settings.b; %batch size
nb = floor(nf/b);
for j=1:nb
    batches{j} = ((j-1)*b+1):(j*b); %disjoint
end
settings.numBatches = nb;
%% generate measurement operators
for t=1:nf
    %kmask = kmasks{t};
    kmask = squeeze(kdata(:,:,t,1))~=0;
    Q = (1/sqrt(nx*ny))*Gdft('ifftshift', 1, 'fftshift', 1, 'samp', kmask);
    S = cell(nc,1);
    F = cell(nc,1);
    for j=1:nc
        S{j} = Gdiag(b1(:,:,j));
        F{j} = Q;
    end
    S = block_fatrix(S, 'type', 'col');
    F = block_fatrix(F, 'type', 'diag');
    A{t} = F*S; %measurement operator for ith frame
end
clear Q S F
%% batch input data
clear X;
for t=1:nf
    kmask = squeeze(kdata(:,:,t,:))~=0;
    btmp = squeeze(kdata(:,:,t,:));    
    B{t} = btmp(kmask);
end
%% initialize low-rank subspace
dims = settings.dims;
kmask_sum = zeros(dims);
for t=1:nf
    %kmask = kmasks{t};
    kmask = squeeze(kdata(:,:,t,1))~=0;
    kmask_sum = kmask_sum + kmask;
end
%find common k-space lines
kmask_common = (kmask_sum/nf > 0.7);
%% two-stage subspace init
fprintf("running two=stage subspace initialization\n");
tic;
for t=1:nf
    kmask = kmask_common;
    Q = (1/sqrt(nx*ny))*Gdft('ifftshift', 1, 'fftshift', 1, 'samp', kmask);
    S = cell(nc,1);
    F = cell(nc,1);
    for j=1:nc
        S{j} = Gdiag(b1(:,:,j));
        F{j} = Q;
    end
    S = block_fatrix(S, 'type', 'col');
    F = block_fatrix(F, 'type', 'diag');
    Atmp{t} = F*S; %measurement operator for ith frame
end
clear Q S F
sos = sum(abs(b1).^2,3);
r = settings.r;
Xinit = zeros(prod(dims),nf);
for t=1:nf
    btmp = squeeze(kdata(:,:,t,:));
    mask = repmat(kmask_common,[1 1 nc]);
    zf = reshape(Atmp{t}'*btmp(mask),dims)./sos;
    Xinit(:,t) = zf(:);
end
[Uj,~,~] = svds(Xinit,r);

% grouse refinement of initialization
sos = sum(abs(b1).^2,3);
sos = sos(:);
Ut = Uj;
for t = [randperm(nf)]%t=[nf:-1:1]
    % compute gradient, projection, and residual
    xt = B{t};
    Qt = A{t}*Ut;
    wt = Qt\xt;
    pt = Ut*wt;
    rt = A{t}'*((A{t}*pt)-xt);
    %reweight residual?
    rt = rt./sos;

    % greedy stepsize
    theta = norm(rt)*norm(pt)*0.00001;

    % subspace update
    Ut = Ut + ((cos(theta)-1)*(pt/norm(pt,2)) - sin(theta)*(rt/norm(rt,2)))*(wt'/norm(wt,2));
end
U0 = Ut;
toc;
%% show subspace initalization 
figure(1); im(permute(reshape(abs(U0),[dims,r]),[2 1 3])); title('subspace initialization');
%% grouse recon
fprintf("running GROUSE");
Ut = U0;
tic;
for t=nf:-1:1
    % compute gradient, projection, and residual
    xt = B{t};
    Qt = A{t}*Ut;
    wt = Qt\xt;
    pt = Ut*wt;
    rt = A{t}'*((A{t}*pt)-xt);
    
    %reweight residual
    rt = rt./sos;

    % greedy stepsize
    theta = atan(norm(rt,2)/norm(pt,2));

    % subspace update
    Ut = Ut + ((cos(theta)-1)*(pt/norm(pt,2)) - sin(theta)*(rt/norm(rt,2)))*(wt'/norm(wt,2));

    Qt = A{t}*Ut;
    wt = Qt\xt;
    recon1(:,:,t) = reshape(Ut*wt,dims);
end
toc;
%% show initial GROUSE recon
figure(2); im(permute(recon1,[2 1 3])); title('GROUSE recon');
%% set RUFFed GROUSE parameters
nb = length(batches);
cost = [];
lam = 5e-3; %settings.lam;
niters = 50;%settings.pd.niters;
sigma = 0.1;%settings.pd.sigma; %dual step-size
tau = 1;%settings.pd.tau; %primal step-size 
[T,Tt] = temporal_fd(settings);
sos = sum(abs(b1).^2,3);
sos = sos(:);
eta = 0.0002;%0.0005;
%% run RUFFed GROUSE
Uj = U0;
tic;
for j = 1:nb %iterate   over all batches  
    %initalize variables for current batch
    batch = batches{j};
    b = length(batch);
    Bj = B(batch); %measurement vec for this batch
    Aj = A(batch); %measurement ops for this batch
    for i = 1:b
        Q{i} = Aj{i}*Uj;    %compressed subspace
        [QUi,~,~] = svd(Q{i},'econ');
        QU{i} = QUi; %projector onto compressed subspace
    end
    
    %primal-dual solution of sparse residual
    Sj = zeros([prod(dims),b]); %sparse residual
    Yj = zeros(size(T(Sj)));    %dual variable
    for k=1:niters    
        %sparse update
        TtYj = Tt(Yj);
        for i=1:b
            AS{i} = Aj{i}*Sj(:,i);
            diff = AS{i}-Bj{i};
            pdiff{i} = diff-QU{i}*(QU{i}'*diff); %orthoprojection
            Sj(:,i) = Sj(:,i) - tau*((Aj{i}'*pdiff{i})./sos+TtYj(:,i));
        end
        
        %dual update
        TSj = T(Sj);
        Yj = proj_infty(Yj+sigma*TSj,lam);
                    
        datafit = 0;
        for i=1:b
            datafit = datafit+norm(pdiff{i})^2;
        end
        cost(k) = 0.5*datafit + lam*norm(TSj(:),1);
    end
    %figure(100);
    %plot(cost);
    
    % Compute weights and residual
    for i=1:b
        Wj(:,i) = Q{i}\(Bj{i}-AS{i}); %weights matrix
        QW{i} = Q{i}*Wj(:,i); %compressed low-rank component
        Rj(:,i) = (Aj{i}'*(QW{i} + AS{i} - Bj{i}))./sos;
    end
    
    %L+S reconstruction of j-th frame (vectorized)
    recon{j} = Uj*Wj+Sj;
    
    %subspace update
    grad = Rj*Wj'; 
    [Utilde,Sig,V] = svd(grad,'econ');
    sv = diag(Sig);
    Uj = Uj*V*diag(cos(sv*eta))*V' + Utilde*diag(sin(sv*eta))*V';
    
    %plot frame by frame recon - comment out for speed
    if 0
    for i = 1:b %for each frame in batch 
        wt = abs(Wj(:,i));
        S = reshape(Sj(:,i),dims);
        LplusS = reshape(recon{j}(:,i),dims);
        truth = f(:,:,batch(i));
        L = reshape(Uj,[dims,r]);
        grouserecon = recon1(:,:,batch(i));
        
        % Plotting
        fig = figure(1);
        set(fig,'name',sprintf('Batch %d/%d, Frame %d/%d',j,nb,i,b));

        subplot(2,3,1); imshow(abs(truth),[0,1]); colormap(sqrt(gray));    
        title('truth');

        subplot(2,3,2); imshow(abs(LplusS),[0,1]);  colormap(sqrt(gray));  
        title('L+S recon');

        subplot(2,3,3); imshow(abs(LplusS-truth),[0,0.2]); 
        title('error, scale: [0,0.2]');

        subplot(2,3,4); imshow(abs(L(:,:,1)),[]);  colormap(sqrt(gray));  
        xlabel(sprintf('weight=%2.1f',wt(1)));
        title('low-rank component L(:,1)'); 

        if size(L,3) > 1
            subplot(2,3,5); imshow(abs(L(:,:,2)),[]);  colormap(sqrt(gray));  
            xlabel(sprintf('weight=%2.1f',wt(2)));
            title('low-rank component L(:,2)');   
        end

        subplot(2,3,6); imshow(abs(S),[]); 
        title('sparse component S');

        drawnow;
        pause(0.5);
    end
    end
end
runtime2 = toc;
fprintf('runtime=%4.2f\n',runtime2);

%reshape output
recon2= [];
for j=1:nb
    for i=1:b
        recon2(:,:,(1:b)+b*(j-1)) = reshape(recon{j},[dims,b]);
    end
end
%% show RUFFed GROUSE recon
figure(3); im(permute(recon2,[2 1 3])); title('RUFFed GROUSE recon');