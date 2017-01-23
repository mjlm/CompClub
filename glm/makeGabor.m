function img = makeGabor(nPix, lambda, theta, phase, sigma, gamma)
% img = makeGabor(nPix, lambda, theta, phase, sigma, gamma) creates a Gabor
% filter as described in https://en.wikipedia.org/wiki/Gabor_filter.
% 
% Inputs:
% nPix      Edge length of patch, in pixels.
% lambda    Wavelength of Gabor (as fraction of patch size).
% theta     Orientation (in radians).
% phase     Phase offset of the Gabor.
% sigma     Standard deviation of the Gaussian envelope (as fraction of patch size).
% gamma     Spatial aspect ratio of the Gabor.
%
% Output:
% img       A nPix-by-nPix matrix containing the Gabor values.
%
% 1/23/2017 Matthias Minderer

if nargin<1
    nPix = 16;
end
if nargin<2
    lambda = 0.5;
end
if nargin<3
    theta = rand*2*pi;
end
if nargin<4
    phase = 0;
end
if nargin<5
    sigma = 1/6;
end
if nargin<6
    gamma = 1;
end

% Express spatial variables in terms of pixels:
lambda = lambda * nPix;
sigma = sigma * nPix;

% Gabor as described in https://en.wikipedia.org/wiki/Gabor_filter:
[xGrid, yGrid] = meshgrid((1:nPix)-nPix/2);
x_ = xGrid .* cos(theta) + yGrid .* sin(theta);
y_ = -xGrid .* sin(theta) + yGrid .* cos(theta);
img = exp(-(x_.^2 + gamma.^2 .* y_.^2)./(2.*sigma.^2)) .* ...
    sin(2.*pi.*x_/lambda + phase);