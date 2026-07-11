%% Set directory
saveDirectory = '/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/Beta_NRas/Sample5/Mito/2025-04-18/nano_004_G/Segment/res/segment_ALSM2';

% Specify the cell IDs to process
cellList = [];   % e.g. [1 2 3 4]

% Segmentation file
fn = '/endosome/archive/bioinformatics/Danuser_lab/Dean/Shen/Liver/1stHCC_Kelly_betacatin_NRas/Beta_NRas/Sample5/Mito/2025-04-18/nano_004_G/Segment/res/segment_ALSM2/uSegment3D_ruffle_labels_postprocess-diffuse_labels.tif';

%% Calculate and save global geometry features
for n = 1:length(cellList)

    fprintf('Processing Cell %d...\n', cellList(n));

    % Load segmented image
    image3D = load3DImage( ...
        fullfile(saveDirectory, ['Cell' num2str(cellList(n))], 'thresholded'), ...
        fn);

    % Output folder
    saveCellPath = fullfile(saveDirectory, ...
        ['Cell' num2str(cellList(n))], ...
        'GlobalMorphology');

    if ~exist(saveCellPath, 'dir')
        mkdir(saveCellPath);
    end

    % Calculate morphology features
    [globalGeoFeature, convexImage, Image] = ...
        calGlobalGeometricFeature(image3D);

    % Save results
    save(fullfile(saveCellPath, 'globalGeoFeature.mat'), 'globalGeoFeature');
    save(fullfile(saveCellPath, 'convexImage.mat'), 'convexImage');
    save(fullfile(saveCellPath, 'Image.mat'), 'Image');

end

fprintf('Finished processing all cells.\n');