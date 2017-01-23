function [frac, D_model, D_null] = getDeviance(y, yHat, mean_y_train, family)
% [frac, D_model, D_null] = getDeviance(y, yHat, mean_y_train, family)
% calculates the deviance for a variety of models.
%
% Inputs:
% y                 Data 
% yHat              Model prediction
% mean_y_train      Mean of the training set of y (ensures that the null model does not get access to the test set).
% family            Model distribution. "Poisson" and "Gaussian" are currently implemented.
%
% Outputs:
% frac              The fraction of the null-model deviance that is explained by the model.
% D_model           The residual deviance of the model.
% D_null            The residual deviance of the null model, i.e. a model with only one free parameter (the mean of the data).

if nargin < 4
    family = 'Poisson';
end

if nargin < 3 || isempty(mean_y_train)
    mean_y_train = mean(y);
end

y = y(:);
mu = yHat(:);

switch family
    case 'Poisson'
        % Some useful sources:
        % https://en.wikipedia.org/wiki/Deviance_(statistics)
        % http://thestatsgeek.com/2014/04/26/deviance-goodness-of-fit-test-for-poisson-regression/
        % http://stats.stackexchange.com/questions/15730/poisson-deviance-and-what-about-zero-observed-values
            
        D_model = 2 * sum(nanRep(y .* log(y ./ mu), 0) + mu - y);
        D_null = 2 * sum(nanRep(y .* log(y ./ mean_y_train), 0) + mean_y_train - y);
        
    case 'Gaussian'
        % This is simply the R-squared, https://en.wikipedia.org/wiki/Coefficient_of_determination
        D_model = sum((y-mu).^2);
        D_null = sum((y-mean_y_train).^2);
end

frac = 1 - D_model/D_null;