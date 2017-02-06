%{
Comp Club: Generalized Linear Models

This code goes along with the Comp Club sessions held on 1/23/17 and 2/6/17
by Selmaan, Laura and Matthias.

---------------------------------------------------------------------------

Regularization basics:

This script builds on the glmCompClub_basicRegression.m script. It
demonstrates the problem of overfitting to show why cross-validation is
important, and shows how to use regularization to reduce overfitting.
%}

%% Set up:
clear
close all

%% Create artificial data to simulate a V1 simple cell:
% We first create a receptive field for our artificial neuron, and then
% simulate the response of a neuron with this receptive field to a
% white-noise visual stimulus. If this were a real experimet, we wouldn't
% know this receptive field but would have to estimate it using regression.
% That's what we'll do below.

% Create receptive field:
rfTrue = makeGabor; % Have a look at the makeGabor.m file for a description.
figure(1)
imagesc(rfTrue)
axis off
axis equal
title('The true (unknown) receptive field')

% Create stimulus:
nFrames = 500;
stimStd = 0.3;
s = randn(size(rfTrue, 1), size(rfTrue, 2), nFrames) * stimStd;
figure(2)
imagesc(s(:,:,1))
colormap gray
axis off
axis equal
title('One frame of the stimulus movie')

% Simulate response:
r = simulatedNeuron(rfTrue, s); % Check the function file to learn how the simulator works.
figure(3)
plot(r)
xlabel('Frame')
ylabel('Simulated firing rate')
title('The simulated neural recording')

%% Use linear regression to find the receptive field (STA):
X = reshape(s, [], nFrames)';
X = cat(2, ones(500, 1), X); % Add bias term.

% This is the classic Spike-Triggered Average (STA). More details:
% https://en.wikipedia.org/wiki/Spike-triggered_average
sta = (X' * X) \ (X' * r);

% Predict response and calculate explained variance:
rHatSta = X * sta;
varianceExplained = corr(r, rHatSta)^2;

figure(4)
imagesc(reshape(sta(2:end), size(rfTrue)))
axis off
axis equal
title('Spike-Triggered Average')

%% Use log-Poisson Generalized Linear Model to find the receptive field:
X = reshape(s, [], nFrames)';

rfGlm = glmfit(X, r, 'Poisson');

% Predict response and calculate explained deviance:
rHatGlm = exp(X * rfGlm(2:end) + rfGlm(1));
devianceExplained = getDeviance(r, rHatGlm);

figure(5)
imagesc(reshape(rfGlm(2:end), size(rfTrue)))
axis off
axis equal
title('GLM-estimate of RF')

%% More realistic scenario: Add some noise to the data:
snr = 1;
rNoisy = simulatedNeuron(rfTrue, s, snr); % Check the function file to learn how the simulator works.

%% Fit GLM to noisy data:
X = reshape(s, [], nFrames)';

rfGlmNoisy = glmfit(X, rNoisy, 'Poisson');

% Predict response and calculate explained deviance:
rHatGlmNoisy = exp(X * rfGlmNoisy(2:end) + rfGlmNoisy(1));
devianceExplainedNoisy = getDeviance(rNoisy, rHatGlmNoisy);

figure(6)
imagesc(reshape(rfGlmNoisy(2:end), size(rfTrue)))
axis off
axis equal
title({'GLM-estimate on noisy data', ...
    sprintf('Explained deviance: %1.1f%% !?', 100*devianceExplainedNoisy)})

%% How can it explain so much variance if the RF looks so crappy???
% Enter Laura.

%% How to detect overfitting: Separate training and test sets.

% To detect overfitting we split the data into a training set and a test
% set: The model is trained on one part of the data, and then tested on a
% withheld second part of the data. If the model is too flexible and fit
% noise in the training set, it will generalize poorly and have a high test
% error.

% Typically, 1/5 or 1/10 of the data is reserved for testing. Note that for
% timeseries data, adjacent datapoints are typically highly correlated. So
% the train and test split should be done in large chunks, rather than
% individual datapoints. Otherwise, the training and test set might be
% highly correlated, which defeats their purpose.

% For simplicity, here we just split the data in half:
isTrain = (1:nFrames) <= nFrames/2;
isTest = ~isTrain;

%% GLM fit with test error:
X = reshape(s, [], nFrames)';

rfGlmNoisyTrain = glmfit(X(isTrain, :), rNoisy(isTrain), 'Poisson');
bias = rfGlmNoisyTrain(1);
rfGlmNoisyTrain = rfGlmNoisyTrain(2:end);

% Predict response and calculate explained deviance:
rHatGlmNoisyTrain = exp(X(isTrain, :) * rfGlmNoisyTrain + bias);
rHatGlmNoisyTest = exp(X(isTest, :) * rfGlmNoisyTrain + bias);
devianceExplainedNoisyTrain = getDeviance(rNoisy(isTrain), rHatGlmNoisyTrain);
devianceExplainedNoisyTest = getDeviance(rNoisy(isTest), rHatGlmNoisyTest);

figure(7)
imagesc(reshape(rfGlmNoisyTrain, size(rfTrue)))
axis off
axis equal
title({'GLM-estimate with training/test split', ...
    sprintf('Dev train=%1.1f%% test=%1.1f%%', ...
    100*devianceExplainedNoisyTrain, 100*devianceExplainedNoisyTest)})

%% GLM fit on noisy data with regularization:
X = reshape(s, [], nFrames)';

% Regularization parameters:
lambda = 0.01; % Regularization strength. 0.01 is a good start here.
alpha = 0.01; % Trades off between L2 and L1 regularization (alpha of 0 is pure L2).

tic
[rfRegularized, fitInfo] = lassoglm(X(isTrain, :), rNoisy(isTrain), ...
    'poisson', 'Lambda', lambda, 'Alpha', alpha);
timeToFitOneLambda = toc;
bias = fitInfo.Intercept;

% Predict response and calculate explained deviance:
rHatRegularizedTrain = exp(X(isTrain, :) * rfRegularized + bias);
rHatRegularizedTest = exp(X(isTest, :) * rfRegularized + bias);
devianceExplainedRegTrain = getDeviance(rNoisy(isTrain), rHatRegularizedTrain);
devianceExplainedRegTest = getDeviance(rNoisy(isTest), rHatRegularizedTest);

figure(8)
imagesc(reshape(rfRegularized, size(rfTrue)))
axis off
axis equal
title({sprintf('GLM with regularization (l=%1.2f, a=%1.2f)', lambda, alpha), ...
    sprintf('Dev train=%1.1f%% test=%1.1f%%', ...
    100*devianceExplainedRegTrain, 100*devianceExplainedRegTest)})

%% But how do we find the best lambda value to use?
% We simply try a range of different values and see where the error is
% lowest. We use cross-validation to get the error estimate, so that we are
% not fooled by over-fitting.
[rfCvAll, fitInfo] = lassoglm(X(isTrain, :), rNoisy(isTrain), ...
    'poisson', 'CV', 10, 'Alpha', alpha);

figure(9)
lassoPlot(rfCvAll, fitInfo, 'plottype', 'CV', 'parent', gca);
rfCv = rfCvAll(:, fitInfo.Index1SE);
rHatCvTest = exp(X(isTest, :) * rfCv + fitInfo.Intercept(fitInfo.Index1SE));
devianceExplainedCvTest = getDeviance(rNoisy(isTest), rHatCvTest);

figure(10)
imagesc(reshape(rfCv, size(rfTrue)))
axis off
axis equal
title({sprintf('GLM with optimal regularization (l=%1.3f)', fitInfo.Lambda(fitInfo.Index1SE)), ...
    sprintf('Dev test=%1.1f%%', 100*devianceExplainedCvTest)})

%% Show regularization path and compare with Lasso (pure L1)
figure(11)
plot(log(fitInfo.Lambda), rfCvAll', 'linewidth', 3)
title( 'L2 (ridge) regularization path')
xlabel('Regulatization strength (log(lambda))')
ylabel('Coefficient values')

% Compare with pure lasso (L1) regularization:
[rfCvAllLasso, fitInfoLasso] = lassoglm(X(isTrain, :), rNoisy(isTrain), ...
    'poisson', 'CV', 10, 'Alpha', 1);

figure(12)
rfCvLasso = rfCvAllLasso(:, fitInfoLasso.Index1SE);
im = imagesc(reshape(rfCvLasso, size(rfTrue)));
im.AlphaData = reshape(rfCvLasso, size(rfTrue))~=0;
axis off
axis equal
title('GLM with pure L1 (lasso) regularization')

figure(13)
plot(log(fitInfoLasso.Lambda), rfCvAllLasso', 'linewidth', 3)
title( 'L1 (lasso) regularization path')
xlabel('Regulatization strength (log(lambda))')
ylabel('Coefficient values')









