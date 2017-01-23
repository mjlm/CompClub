%{
Comp Club: Generalized Linear Models

This code goes along with the Comp Club sessions held on 1/23/17 by
Selmaan, Laura and Matthias.

---------------------------------------------------------------------------

Regression basics:

This script goes over the basics of fitting a regression model to relate
stimuli or behavioral data to neural activity.

IMPORTANT: Please be aware that this script only contains a minimal example
and lacks several aspects that are absolutely essential for any real data
analysis, such as cross-validation and regularization. Those topics are
discussed in the glmCompClub_regularization.m script.
%}

%% Set up:
clear
close all
h = []; % A variable to hold figure handles.

% Tip: This is a very helpful non-default Matlab setting:
% Home-->Preferences-->Editor/Debugger-->Display-->Enable datatips in edit
% mode

%% Load data:
% The data come from posterior parietal cortex of a mouse performing a
% simple task in a virtual environment. The mouse gets rewards for running
% forward. There is no fixed trial structure. The neural data was acquired
% using two-photon imaging of GCaMP6s and deconvolved to yield an estimate
% of spike rate (in arbitrary units).
matfile = load('data.mat');

% Deconvolved calcium fluorescence:
firingRate = matfile.firingRate;

% Features of mouse running behavior:
% We standardize (zscore) the data to remove arbitrary scaling:
forwardVelocity = zscore(matfile.forwardVelocity);
angularVelocity = zscore(matfile.angularVelocity);
forwardAcceleration = zscore(matfile.forwardAcceleration);
angularAcceleration = zscore(matfile.angularAcceleration);

% Timebase:
samplingRate = matfile.samplingRate;
t = (1:numel(firingRate)) / samplingRate;

% Inspect data:
h.fig.data = figure(1);
clf
h.ax.behavior = subplot(2, 1, 1);
plot(t, forwardVelocity)
ylabel('Forward velocity')

h.ax.firingRate = subplot(2, 1, 2);
plot(t, firingRate, 'displayname', 'True firing rate')
ylabel('Estimated firing rate')
xlabel('Time (seconds)')
linkaxes([h.ax.behavior, h.ax.firingRate], 'x')

%% Simple linear regression (single explanatory variable):
% Create data matrix. Should be nTimepoints-by-nFeatures:
X = forwardVelocity;

% To model a baseline offset, we augment the data with a column of ones:
X = cat(2, ones(size(X, 1), 1), X);

% Response vector should be nTimepoints-by-1:
y = firingRate;

% Perform regression using the Normal Equation:
betaReg = (X'*X) \ (X'*y);

% Get model prediction of firing rate:
firingRatePredictedReg = X * betaReg;

% Get the model fit quality under the assumption of Gaussian noise, i.e.
% the coefficient of variation (a.k.a. "R squared"). This is a special case
% of the concept of "deviance" for Gaussian distributions. For more info,
% see https://en.wikipedia.org/wiki/Coefficient_of_determination:
varianceExpl = corr(y, firingRatePredictedReg)^2;

% Plot regression line:
h.fig.scatter = figure(2);
clf
h.ax.scatter = axes;
hold on
firingRateJittered = firingRate + randn(size(firingRate)) * 0.05;
plot(forwardVelocity, firingRateJittered, '.', 'markersize', 1, ...
    'displayname', 'Data')
plot(forwardVelocity, firingRatePredictedReg, ...
    'displayname', 'Linear regression fit')
xlabel('Running velocity')
ylabel('Estimated firing rate')
legend('show')

% Plot predicted activity:
hold(h.ax.firingRate, 'on')
plot(h.ax.firingRate, t, firingRatePredictedReg,  ...
    'displayname', sprintf('Simple lin. reg: %1.1f%% expl.', 100*varianceExpl));
legend(h.ax.firingRate, 'Show')

%% Log-Poisson Generalized Linear Model:

% GLMs with distributions other than the Gaussian do not have a nice,
% elegant analytical solution and are fit usign some sort of optimization
% algorithm. Thankfully, MATLAB has a function for this. We set 'constant'
% to off, indicating that we do not want the function to a constant term
% (column of ones), because we did that ourselves above.
betaGlm = glmfit(X, y, 'poisson', 'constant', 'off');

% Get prediction:
firingRatePredictedGlm = exp(X * betaGlm);

% Get fraction of explained deviance. See getDeviance() for details:
devianceExpl = getDeviance(y, firingRatePredictedGlm);

% Plot regression line:
plot(h.ax.scatter, forwardVelocity, firingRatePredictedGlm, ...
    'displayname', 'GLM fit')
legend(h.ax.scatter, 'off')
legend(h.ax.scatter, 'show')

% Plot predicted activity:
hold(h.ax.firingRate, 'on')
plot(h.ax.firingRate, t, firingRatePredictedGlm,  ...
    'displayname', sprintf('Simple GLM: %1.1f%% expl.', 100*devianceExpl));
legend(h.ax.firingRate, 'off')
legend(h.ax.firingRate, 'show')

%% Multiple explanatory variables:
% Any number of features can combined in the columns of the data matrix:
X = cat(2, ...
    ones(size(X, 1), 1), ...
    forwardVelocity, ...
    angularVelocity, ...
    forwardAcceleration, ...
    angularAcceleration);

betaGlmMulti = glmfit(X, y, 'poisson', 'constant', 'off');
firingRatePredictedGlmMulti = exp(X * betaGlmMulti);
devianceExplMulti = getDeviance(y, firingRatePredictedGlmMulti);

% Plot predicted activity:
hold(h.ax.firingRate, 'on')
plot(h.ax.firingRate, t, firingRatePredictedGlmMulti,  ...
    'displayname', sprintf('Multiple GLM: %1.1f%% expl.', 100*devianceExplMulti));
legend(h.ax.firingRate, 'off')
legend(h.ax.firingRate, 'show')

% Inspect model coefficients to understand the influence of different
% explanatory variables on the prediction:
figure(3)
bar(betaGlmMulti);
set(gca, 'xticklabels', {'Offset', 'Forward vel.', 'Angular vel.', ...
    'Forward acc.', 'Angular acc.'}, 'XTickLabelRotation', -45)
ylabel('Coefficient value')
title('GLM coefficients')
