function [lik, latents] = likfun_drift(x,data)
    
    % Likelihood function for Go-NoGo task with different starting points
    % of the DDM for "escape" and "avoid" conditions.
    %
    % USAGE: [lik, latents] = likfun_start(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - non-decision time (T)
    %       x(2) - drift rate general go bias weight (b0)
    %       x(3) - drift rate differential action value weight (b1)
    %       x(4) - drift rate Pavlovian bias, escape condition (b2)
    %       x(5) - drift rate Pavlovian bias, avoid condition (b3)
    %       x(6) - learning rate for state-action values (alpha)
    %       x(7) - decision threshold (omega)
    %   data - structure with the following fields
    %           .c - [N x 1] choices
    %           .r - [N x 1] rewards
    %           .Escape - [N x 1] escape condition (1) avoid condition (0)
    %           .C - number of choice options
    %           .N - number of trials
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %   latents - structure with the following fields:
    %           .v - [N x 1] drift rate
    %           .pGo - [N x 1] probability of Go action
    %           .RT_mean - [N x 1] mean response time
    %
    % Sam Gershman, June 2017
    
    % set parameters
    T = x(1);
    b1 = x(2);
    b2 = x(3);
    b3 = x(4);
    w = x(5);
    lr = x(6);
    omega = x(7);
    
    % initialization
    Tmax = 2; % response epoch
    lik = 0;
    Q = zeros(4,2);    % state-action (instrumental) values
    data.RT = max(eps,data.RT - T);
    
    for n = 1:data.N
        
        % data for current trial
        c = data.c(n);
        s = data.cond(n);
        
        % drift rate
        v = b1*data.Escape(n) + b2*(1-data.Escape(n)) + b3*(Q(s,2)-Q(s,1));
        
        % accumulate log-likelihod
        if c == 1
            P = wfpt(data.RT(n),-v,omega,1-w);          % Go
        else
            P = integral(@(t) wfpt(t,v,omega,w),0,Tmax-T); % NoGo
        end
        if isnan(P) || P==0; P = realmin; end
        lik = lik + log(P);
        
        % update values
        Q(s,c+1) = Q(s,c+1) + lr*(data.r(n) - Q(s,c+1));
        
        % store latent variables
        if nargout > 1
            latents.v(n,1) = v;
            latents.pGo(n,1) = integral(@(t) wfpt(t,-v,omega,1-w),0,Tmax-T);
            t = linspace(0.001,Tmax-T,100);
            p = wfpt(t,-v,omega,1-w);
            p=p./sum(p);
            latents.RT_mean(n,1) = t*p' + T;
        end
        
    end