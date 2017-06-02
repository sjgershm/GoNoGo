function [results, bms_results] = fit_models(data,results,models)
    
    % Fit GoNoGo models using mfit (https://github.com/sjgershm/mfit).
    %
    % USAGE: [results, bms_results] = fit_models(data,results,models)
    %
    % INPUTS:
    %   data - [S x 1] structure array of data for S subjects
    %
    % OUTPUTS:
    %   results - [M x 1] model fits
    %   bms_results - Bayesian model selection results
    %
    % Sam Gershman, June 2017
    
    likfuns = {'likfun_start' 'likfun_drift'};
    M = length(likfuns);
    if nargin < 1; load data; end
    if nargin < 2; results = []; end
    if nargin < 3; models = 1:M; end
    
    for m = models
        disp(['... fitting model ',num2str(m),' out of ',num2str(M)]);
        
        switch likfuns{m}
            
            case {'likfun_drift'}
                
                param(1).name = 'T';
                param(1).hp = [];
                param(1).logpdf = @(x) 0;
                param(1).lb = -0.2; % lower bound
                param(1).ub = 0.5; % upper bound
                
                param(2).name = 'b1';
                param(2).hp = [];
                param(2).logpdf = @(x) 0;
                param(2).lb = -20; % lower bound
                param(2).ub = 20;   % upper bound
                
                param(3).name = 'b2';
                param(3).hp = [];
                param(3).logpdf = @(x) 0;
                param(3).lb = -20; % lower bound
                param(3).ub = 20;   % upper bound
                
                param(4).name = 'b3';
                param(4).hp = [];
                param(4).logpdf = @(x) 0;
                param(4).lb = -20; % lower bound
                param(4).ub = 20;   % upper bound
                
                param(5).name = 'w';
                param(5).hp = [];
                param(5).logpdf = @(x) 0;
                param(5).lb = 0.001; % lower bound
                param(5).ub = 0.999;   % upper bound
                
                param(6).name = 'alpha';
                param(6).logpdf = @(x) 0;
                param(6).lb = 0;
                param(6).ub = 1;
                
                param(7).name = 'omega';
                param(7).logpdf = @(x) 0;
                param(7).lb = 1e-3;
                param(7).ub = 20;
                
            case {'likfun_start'}
                
                param(1).name = 'T';
                param(1).hp = [];
                param(1).logpdf = @(x) 0;
                param(1).lb = -0.2; % lower bound
                param(1).ub = 0.5; % upper bound
                
                param(2).name = 'b0';
                param(2).hp = [];
                param(2).logpdf = @(x) 0;
                param(2).lb = -20; % lower bound
                param(2).ub = 20;   % upper bound
                
                param(3) = param(2);
                param(3).name = 'b1';
                
                param(4).name = 'w1';
                param(4).hp = [];
                param(4).logpdf = @(x) 0;
                param(4).lb = 0.001; % lower bound
                param(4).ub = 0.999;   % upper bound
                
                param(5) = param(4);
                param(5).name = 'w2';
                
                param(6).name = 'alpha';
                param(6).logpdf = @(x) 0;
                param(6).lb = 0;
                param(6).ub = 1;
                
                param(7).name = 'omega';
                param(7).logpdf = @(x) 0;
                param(7).lb = 1e-3;
                param(7).ub = 20;
                
        end
        
        fun = str2func(likfuns{m});
        results(m) = mfit_optimize(fun,param,data);
    end
    
    % Bayesian model selection
    if nargout > 1
        bms_results = mfit_bms(results);
    end