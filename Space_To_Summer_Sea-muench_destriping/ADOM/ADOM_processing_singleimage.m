%% TODO
% Change these variables

% search path
current_path = '/Volumes/LaCie/test_images/05_17_2015/seadas';

% name of the image to be processed
original_image_name = '/Volumes/LaCie/test_images/05_17_2015/seadas/seadas_products_cloud_albedo.tif';

% specifies name to save the file as
savefolder = '/Volumes/LaCie/test_images/05_17_2015/ADOM';
savefilename = 'cloud_albedo_stripe_corrected.tif';

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
figure(1)
imshow(green)
title('original')

% no data layer (from cloud coverage and land)
NoData = (green==241); % 241 is value of no-data layer from SeaDAS
green(NoData==1) = 255;

% adjusting contrast
green_adapt = adapthisteq(green);
%green_adapt = green;

% display contrast
figure(2)
imshow(green_adapt)
title('adapthisteq')

%% Green

green_adapt_stripesremoved = RemoveAllStripes(green_adapt,rotation_angle,coords,m,n,NoData,crop_whitespace, original_image_name);
%green_adapt_stripesremoved = green_adapt;

figure
imshow(green_adapt_stripesremoved)
title('final')
%% save images

if save_file == true
    imwrite(green_adapt_stripesremoved,fullfile(savefolder,savefilename));
end

%% Functions

function [final_im]=RemoveAllStripes(im,rotation_angle,coords,m,n,NoData,crop_whitespace, original_image_name)
% remove stripes
im = double(im);
% rotating to make stripes vertical
rotated = imrotate(im,rotation_angle);
% remove vertical stripes
fprintf('Vertical destriping %s...\n', original_image_name)
stripes_removed1 = xRemoveStripesVertical(rotated);
% remove horizontal stripes
% rotate so that the horizontal stripes are now vertical
rotated2 = imrotate(stripes_removed1,90);
% remove vertical stripes
fprintf('Horizontal destriping %s...\n', original_image_name)
stripes_removed2 = xRemoveStripesVertical(rotated2);
% final image
% rotating to original orientation and cropping black space from imrotate
% rotate back to original orientation and convert to uint8
final_im = uint8(imrotate(stripes_removed2,-rotation_angle-90));

% crop out blackspace created by imrotate
% might need to adjust coordinates
if crop_whitespace == true
    final_im = imcrop(final_im, [coords(1) coords(2) n-1 m-1]);
    % get rid of edge effects from filtering
    final_im(NoData==1)=255;
end
end

function [output,iteration] = xRemoveStripesVertical(Is)
    % parameters
    opts.tol=1.e-4;
    opts.maxitr=30;

    % case 3, 5
%     opts.beta1=5;
%     opts.beta2=5;
%     opts.beta3=5;
%     opts.lambda1=1.e-1;
%     opts.lambda2=1.e-1;
%     opts.lambda2=1.e-1+5.e-2;

    % case 1, 2, 4
    opts.beta1=1;
    opts.beta2=1;
    opts.beta3=1;
    opts.lambda1=1.e-2;
    opts.lambda2=1.e-2;

    opts.limit=1;

    if max(Is,[],'all')>3000
        %peakval=4096;
        %peakval=16384;
        peakval=8192; % paviau_b103;
        %peakval=32768;
        %peakval=65536;
    elseif max(Is,[],'all')>512
        peakval=2048;
    else
        peakval=256;
    end

    [StripeComponent,iteration]=adom(Is/peakval,opts);

    output=Is/peakval-StripeComponent;
end
