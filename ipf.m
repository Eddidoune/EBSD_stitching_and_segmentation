%% Import Script for EBSD Data
%
% This script was automatically created by the import wizard. You should
% run the whoole script or parts of it in order to import your data. There
% is no problem in making any changes to this script.

%% Specify Crystal and Specimen Symmetries
addpath '/home/caroneddy/MATLAB/mtex-5.10.0'
startup_mtex
mtexdata forsterite

% crystal symmetry
CS = {... 
  'notIndexed',...
  crystalSymmetry('m-3m', 'mineral', 'Iron bcc (old)', 'color', [0.53 0.81 0.98]),...
  crystalSymmetry('m-3m', 'mineral', 'Iron fcc', 'color', [0.56 0.74 0.56])};

% plotting convention
setMTEXpref('xAxisDirection','east');
setMTEXpref('zAxisDirection','inPlane');

% which files to be imported
fname = ['/home/caroneddy/These/Python/EBSD/EBSD_Github/2_EBSD_BIG.ctf'];

% create an EBSD variable containing the data
ebsd1 = EBSD.load(fname, 'CS', CS,'interface','ctf','convertEuler2SpatialReferenceFrame');
ebsd1_r = rotate(ebsd1,rotation.byAxisAngle(xvector,180*degree))

grains_1 = calcGrains(ebsd1_r('indexed'),'angle',10*degree)
grains_1(grains_1.grainSize<64) = []
                                    
plot(grains_1, grains_1.meanOrientation)

plot(grains_1, grains_1.boundary)