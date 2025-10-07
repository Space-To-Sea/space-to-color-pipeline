%% TODO
% Change these variables

% search path
current_path = '/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/seadas';

% name of the image to be processed
original_image_name = '/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/seadas/seadas_products_chlor_a_gray_scale.tif';

% specifies name to save the file as
savefolder = '/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/matlab';
savefilename = 'chlor_a_gray_scale_stripe_corrected.tif';

% angle (deg) at which we rotate to make the stripes approximately vertical
rotation_angle = 14.3;

% set to true to crop the whitespace from rotating, false otherwise
crop_whitespace = true;
% set to true to save the image, false otherwise
save_file = true;

% location of the top left corner where the blackspace ends
% the blackspace is created from rotating the image
% and will be cropped out according to these coordinates
coords = [1954 1888];

% parameters for filtering stripes
decNum1 = 12; %10;
decNum2 = 5; %4;
sigma1 = 12; %12;
sigma2 = 5; %5;

%% loading image/set up

% adds pathbases to matlab's searchable directory
addpath(current_path)

% read image
rgb = imread(original_image_name);
green = rgb(:,:,1:3);
green = rgb2gray(green);

% save dimensions
[m,n] = size(green);

% display original
figure
imshow(green)
title('original')

% no data layer (from cloud coverage and land)
NoData = (green==241); % 241 is value of no-data layer from SeaDAS
green(NoData==1) = 255;

% adjusting contrast
green_adapt = adapthisteq(green);
%green_adapt = green;

% display contrast
figure
imshow(green_adapt)
title('adapthisteq')

%% Green

green_adapt_stripesremoved = RemoveAllStripes(green_adapt,rotation_angle,coords,m,n,NoData,decNum1,decNum2,sigma1,sigma2,crop_whitespace);
%green_adapt_stripesremoved = green_adapt;

figure
imshow(green_adapt_stripesremoved)
title('final')
%% save images

if save_file == true
    imwrite(green_adapt_stripesremoved,fullfile(savefolder,savefilename));
end

%% Functions

function [final_im]=RemoveAllStripes(im,rotation_angle,coords,m,n,NoData,decNum1,decNum2,sigma1,sigma2,crop_whitespace)
% remove stripes 
% rotating to make stripes vertical
rotated = imrotate(im,rotation_angle);
% remove vertical stripes
stripes_removed1 = xRemoveStripesVertical(rotated,decNum1,'DB25',sigma1);
stripes_removed1 = uint8(stripes_removed1); % converting double into uint8
% remove horizontal stripes
% rotate so that the horizontal stripes are now vertical
rotated2 = imrotate(stripes_removed1,90);
% remove vertical stripes
stripes_removed2 = xRemoveStripesVertical(rotated2,decNum2,'DB25',sigma2);
stripes_removed2 = uint8(stripes_removed2); % convert from double into uint8
% final image
% rotating to original orientation and cropping black space from imrotate
% rotate back to original orientation
final_im = imrotate(stripes_removed2,-rotation_angle-90);

% crop out blackspace created by imrotate
% might need to adjust coordinates
if crop_whitespace == true
    final_im = imcrop(final_im, [coords(1) coords(2) n-1 m-1]);
    % get rid of edge effects from filtering
    final_im(NoData==1)=255;
end
end

% by Muench et al., 2009
function [nima]=xRemoveStripesVertical(ima,decNum,wname,sigma)

 % wavelet decomposition
 for ii=1:decNum
 [ima,Ch{ii},Cv{ii},Cd{ii}]=dwt2(ima,wname);
 end

 % FFT transform of horizontal frequency bands
 for ii=1:decNum
 % FFT
 fCv=fftshift(fft(Cv{ii}));
 [my,mx]=size(fCv);

 % damping of vertical stripe information
 damp=1-exp(-[-floor(my/2):-floor(my/2)+my-1].^2/(2*sigma^2));
 fCv=fCv.*repmat(damp',1,mx);

 % inverse FFT
 Cv{ii}=ifft(ifftshift(fCv));
 end

 % wavelet reconstruction
 nima=ima;
 for ii=decNum:-1:1
 nima=nima(1:size(Ch{ii},1),1:size(Ch{ii},2));
 nima=idwt2(nima,Ch{ii},Cv{ii},Cd{ii},wname);
 end
 return
end
