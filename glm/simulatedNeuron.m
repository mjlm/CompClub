function r = simulatedNeuron(rf, s, snr, bias)
% r = simulatedNeuron(rf, s, snr) simulates the response R of a
% linear-exponential neuron with the receptive field RF to stimulus S.
% Optionally, Gaussian noise can be added before the nonlinearity to create
% an output with signal-to-noise ratio snr.
%
% Inputs:
%
% rf        Receptive field of the neuron (2D; same size as stimulus).
% s         height-by-width-by-time movie of frames. Same spatial size as rf. 
% snr       Signal-to-noise ratio: var(signal)/var(noise) (see https://en.wikipedia.org/wiki/Signal-to-noise_ratio).
% bias      Constant term of the linear part of the simulated neuron.
%
% Output
% r         Response of the neuron, i.e. exp(stimulus*rf + bias + noise)
%
% % 1/23/2017 Matthias Minderer

if nargin<3
    snr = inf;
end
if nargin<4
    bias = -2;
end


% Reshape stimulus movie from height-by-width-by-time to time-by-nPixels:
nFrames = size(s, 3);
s = reshape(s, [], nFrames)';

% Create noise:
noiseStd = sqrt(var(s(:))/snr);
noise = randn(nFrames, 1) * noiseStd;

% Simulate response:
r = exp(s * rf(:) + bias + noise);