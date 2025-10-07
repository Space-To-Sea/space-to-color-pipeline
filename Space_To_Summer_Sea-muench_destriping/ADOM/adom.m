% This is the official code of ADOM for stripe noise removal in remote sensing images (RSI).
% ADOM: ADMM-Based Optimization Model for Stripe Noise Removal in Remote Sensing Image
% IEEE Access
% 09/25/2023
% Namwon Kim (namwon@korea.ac.kr)

function [s,bag]=adom(o,opts)
    total_time = tic; % Start timing total execution

    fprintf('Initializing variables...\n');
    init_time = tic;
    Dx=defDDxt;  %x-direction finite function
    Dy=defDDyt;  %y-direction finite function
    Dt=defDDt;
    [m,n]=size(o);
    s=zeros(m,n);
    p1=zeros(m,n);
    p2=p1;
    p3=p1;
    groupweight=ones(m,n);
    r=ones(m,n);
    lambda1=opts.lambda1;
    lambda2=opts.lambda2;
    beta1 = opts.beta1;
    beta2 = opts.beta2;
    beta3 = opts.beta3;
    tol=opts.tol;
    maxiter=opts.maxitr;
    gamma = 0.5*(1+sqrt(5));
    fprintf('Initialization took %.4f seconds\n', toc(init_time));

    fprintf('Computing FFT initialization...\n');
    fft_time = tic;
    cir_fft = beta1*(abs(psf2otf([1; -1], [m,n]))).^2 + ...
       beta3*(abs(psf2otf([1, -1], [m,n]))).^2 + beta2;
    fprintf('FFT initialization took %.4f seconds\n', toc(fft_time));

    Dxs=Dx(o-s); % d_x(s)
    Dys=Dy(s); % d_y(s)
    Dx_o=Dx(o);
    a=zeros(m,n);
    b = a;
    c = a;
    bag = zeros(1);
    a1 = 1;
    a2 = 1;
    cal_tolerance=1;
    iter=1;
    sp = s;

    fprintf('\nStarting main iteration loop...\n');
    while cal_tolerance>tol && iter<maxiter
        iter_time = tic;
        fprintf('\nIteration %d:\n', iter);

        % update weights control
        weights_time = tic;
        weight1 = (a2+cal_tolerance+eps)/(a1-cal_tolerance+eps);

        % starting point control
        if iter <= 10
            a1 = a2;
            a2 = (1+sqrt(1+(4*((a1)^2))))/2;
            % limit coefficient
            weightsz=weight1;
        else
            a1 = a2;
            a2 = (1+sqrt(1+(2*((a1)^2))))/2;
            if a1 ~= a2
                weightsz=(a1)/(a2+eps);
            end
        end
        fprintf('Weights update took %.4f seconds\n', toc(weights_time));

        % step size control
        step_time = tic;
        s = s + ((a1-0.1)/(a2))*(s-sp);
        p1 = weightsz*p1;
        p2 = weightsz*p2;
        p3 = weightsz*p3;
        fprintf('Step size control took %.4f seconds\n', toc(step_time));

        V1=Dys+p1/beta1;
        V2=Dxs+p2/beta2;

        % Subproblems
        subprob_time = tic;

        % a-subproblem L1 norm
        a=sign(V1).*max(abs(V1)-1/beta1,0);

        % b-subproblem weighted L1 norm
        b=sign(V2).*max(abs(V2)-(weight1.*lambda1)/beta2,0);
        fprintf('A and B subproblems took %.4f seconds\n', toc(subprob_time));

        % c-subproblem
        % weighted L2,1 norm with contextual information
        % weighted group sparse
        c_time = tic;
        for i=1:n
            % update weights control
            % contextual information
            mean_s = sum(abs(s(:,i)))/n;
            if i==n
                idx=abs(s(:,end-1)-s(:,end))<mean_s;
            elseif i==1
                idx=abs(s(:,i+1)-s(:,i))<mean_s;
            else
                idx=abs(s(:,i+1)-s(:,i))<mean_s | abs(s(:,i-1)-s(:,i))<mean_s;
            end
            groupweight(idx,i) = 1/sqrt((norm(s(:,i)+cal_tolerance,'fro')*2+eps));

            r(:,i)=s(:,i)+p3(:,i)/beta3;
            c(:,i)=r(:,i).*max(norm(r(:,i),'fro')-(groupweight(:,i).*lambda2)/beta3,0)/(norm(r(:,i),'fro')+eps);

        end
        fprintf('C subproblem took %.4f seconds\n', toc(c_time));

        % s-subproblem
        s_time = tic;
        sp=s;
        t1 = beta1*a-p1;
        t2 = beta2*Dx_o-beta2*b+p2;
        t3 = beta3*c-p3;
        sr = Dt(t2,t1)+t3;
        s = real(ifft2(fft2(sr)./(cir_fft+eps)));
        fprintf('S subproblem took %.4f seconds\n', toc(s_time));

        update_time = tic;
        u1=o-s;
        u2=o-sp;

        % update Dxs, Dys
        Dxs=Dx(o-s);
        Dys=Dy(s);

        % update p
        % The golden ratio
        % 1.618 ¡Ö (sqrt(5)+1)/2
        p1=p1+gamma*beta1*(Dys-a);
        p2=p2+gamma*beta2*(Dxs-b);
        p3=p3+gamma*beta3*(s-c);

        % Formal definition (relative error)
        % eta = epsilon/abs(v)=||(v-v_approx)/v||_2
        cal_tolerance = norm(u1-u2,'fro')/norm(u2,'fro');
        bag(iter) = cal_tolerance;
        fprintf('Final updates took %.4f seconds\n', toc(update_time));

        fprintf('Total iteration %d took %.4f seconds\n', iter, toc(iter_time));

        if cal_tolerance< tol
            break;
        end
        iter = iter+1;
    end % loop

    fprintf('\nTotal execution time: %.4f seconds\n', toc(total_time));

end % main

%%%%--------------Subfunction-------------%%%%
function Dx=defDDxt
    Dx=@(U)ForwardD1(U);
end

function Dy=defDDyt
    Dy=@(U)ForwardD2(U);
end

function Dt=defDDt
    Dt= @(X,Y) Dive(X,Y);
end

%    ^^^^
%    |||||...||
%    |||||...||
%    .
%    .
%    .
%    |||||...||
% Input: discrete data
function Dux=ForwardD1(U)
    Dux=[diff(U,1,2),U(:,1)-U(:,end)];
end

%  -----...--
% <
%  -----...--
% .
% .
% .
%  -----...--

%  -----...--
%  -----...--
% .
% .
% .
%  -----...--
function Duy=ForwardD2(U)
    Duy=[diff(U,1,1);U(1,:)-U(end,:)];
end

function DtXY = Dive(X,Y)
    DtXY = [X(:,end) - X(:, 1), -diff(X,1,2)]; % x-directional
    DtXY = DtXY + [Y(end,:) - Y(1, :); -diff(Y,1,1)]; % x-directional + y-directional
end
