%% TODO
% Change these variables

% search path
current_path = '/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/seadas/color';

% name of the image to be processed
original_images_names = {'/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/seadas/color/seadas_products_RGB.tif', '/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/seadas/color/seadas_products_chlor_a_oceancolor.tif', '/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/seadas/color/seadas_products_diatoms_hirata.tif', '/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/seadas/color/seadas_products_dinoflagellates_hirata.tif', '/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/seadas/color/seadas_products_greenalgae_hirata.tif', '/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/seadas/color/seadas_products_prymnesiophytes_hirata.tif'};

% specifies name to save the file as
savefolder = '/Volumes/LaCie/Processing/0910_MATLAB_TEST_BATCH/Processing/07_20_2015/matlab';
savefilenames = {'RGB_stripe_corrected.tif','chlor_a_oceancolor_stripe_corrected.tif','diatoms_stripe_corrected.tif','dinoflagellates_stripe_corrected.tif','greenalgae_stripe_corrected.tif','prymnesiophytes_stripe_corrected.tif'};

% angle (deg) at which we rotate to make the stripes approximately vertical
rotation_angle = 14.3;

% set to true to crop the whitespace from rotating, false otherwise
crop_whitespace = true;
% set to true to save the image, false otherwise
save_file = true;

% location of the top left corner where the blackspace ends
% the blackspace is created from rotating the image
% and will be cropped out according to these coordinates
coords = [1946 1881];

% parameters for filtering stripes
decNum1 = 12;
decNum2 = 4;
sigma1 = 12;
sigma2 = 5;

%% loading image/set up
for i = 1:numel(original_images_names)
    % adds pathbases to matlab's searchable directory
    addpath(current_path)

    % read image
    hi = original_images_names{1};
    disp(hi)
    rgb = imread(original_images_names{i});
    rgb = rgb(:,:,1:3);

    % convert from RGBa (4 channels) to each RGB band (red, green, blue)
    red = rgb(:,:,1);
    green = rgb(:,:,2);
    blue = rgb(:,:,3);

    % save dimensions
    [m,n] = size(green);

    % no data layer (from cloud coverage and land)
    NoData_red = (red==241);
    red(NoData_red==1) = 255;

    NoData_green = (green==241); % 241 is value of no-data layer from SeaDAS
    green(NoData_green==1) = 255;

    NoData_blue = (blue==241);
    blue(NoData_blue==1) = 255;

    % adjusting contrast
    red_adapt = red; %adapthisteq(red);
    green_adapt = green; %adapthisteq(green);
    blue_adapt = blue; %adapthisteq(blue);

    %% Red

    red_adapt_stripesremoved = RemoveAllStripes(red_adapt,rotation_angle,coords,m,n,NoData_red,crop_whitespace);

    %% Green

    green_adapt_stripesremoved = RemoveAllStripes(green_adapt,rotation_angle,coords,m,n,NoData_green,crop_whitespace);
    %% Blue

    blue_adapt_stripesremoved = RemoveAllStripes(blue_adapt,rotation_angle,coords,m,n,NoData_blue,crop_whitespace);
    %% show final

    rgb_final = cat(3,red_adapt_stripesremoved,green_adapt_stripesremoved,blue_adapt_stripesremoved);
    %% save images

    if save_file == true
        imwrite(rgb_final,fullfile(savefolder,savefilenames{i}));
    end
end

%% Functions

function [final_im]=RemoveAllStripes(im,rotation_angle,coords,m,n,NoData,crop_whitespace)
    % remove stripes
    im = double(im);
    % rotating to make stripes vertical
    rotated = imrotate(im,rotation_angle);
    % remove vertical stripes
    stripes_removed1 = xRemoveStripesVertical(rotated);
    % remove horizontal stripes
    % rotate so that the horizontal stripes are now vertical
    rotated2 = imrotate(stripes_removed1,90);
    % remove vertical stripes
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
        opts.maxitr=1000;

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
