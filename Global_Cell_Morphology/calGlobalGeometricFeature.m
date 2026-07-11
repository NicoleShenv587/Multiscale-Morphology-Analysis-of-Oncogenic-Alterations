function [globalGeoFeature convexImage Image] = calGlobalGeometricFeature(Image,varargin)
%calGeoFea calculates the geometric features for an image.
% It uses the regionprop3|regionprop for a 3D|2D image
%
% INPUT image     2D|3D image (matrix) - should be binary image
%
% OUTPUT globalGeoFeature    structure array of global geometric features
% of an
%                     cell image
% convexImage         convex image of the cell (output of regionprop) Image
% image of the cell without blancked frames
%
% Hanieh Mazloom-Farsibaf, Gaudenz Danuser lab, 2020

% ip = inputParser; ip.CaseSensitive = false; ip.addRequired('image');
% ip.addOptional('surface', [1 2 4]);
if nargin > 2
    surface=varargin{1};
    neighbor=varargin{2};
end

%check if it is a segmented image
integerCheck = all((mod(Image(:),1) == 0)); % check if it is se
hasBackground = any(Image(:) == 0) || any(Image(:) == 1);
fewLabelsCheck = numel(unique(Image(:))) < 500;  % less than 500 objects in the treshold as needed
isSegmented = integerCheck && fewLabelsCheck && hasBackground;

if isSegmented > 1
    warning('No binary image was provided. A binary image will be automatically created, please verify the results or provide a binary image.')

    %normalize the image3D first
    Image = imadjustn(Image);
    % use easy threshold to make a binary object
    thresh = multithresh(Image);
    Image (Image <thresh) = 0;
    Image (Image >= thresh) = 1;

    se = strel('sphere',10);

    Image = imclose(Image,se);
    Image = imfill(Image);

    CC = bwconncomp(Image);
    CClength = cellfun(@length, CC.PixelIdxList);
    [  MaxCC MaxCCInd] = max(CClength);
    Image = zeros(size(Image));
    Image(CC.PixelIdxList{MaxCCInd}) = 1;
    Image = single(Image);

    % smoothe the binary image sigma = 1; [image3D] =
    % filterGauss3D(image3D, sigma);
    %or using the smooth3 function in MATLAB
    Image = smooth3(Image);
end

% normalized it to one for more accurate calculation, I tested for [0 255]
% and [0 1], it gives Nan results - it might not be necessary Image =
% single(Image); % to avoid zero when image's class is uint8 or uint16
% Image = (Image - min(Image(:)))/(max(Image(:))- min(Image(:)));

%calculate the basic global features from regionprop3
if size(Image,3) > 1 % 3D image
    s = regionprops3(Image,"ConvexHull","Volume",'ConvexImage',...
        'ConvexVolume','Centroid',"BoundingBox",'Extent','Solidity','Image', ...
        'SurfaceArea','SubarrayIdx','EquivDiameter','PrincipalAxisLength');
    s(s.Volume==0,:)=[]; %to clean up the nonrelevent output of regionprop3
    globalGeoFeature.Volume=s.Volume; % in pixels^3
    globalGeoFeature.SurfaceArea= s.SurfaceArea;
    globalGeoFeature.Solidity= s.Solidity;
    globalGeoFeature.EquivDiameter=s.EquivDiameter;
    globalGeoFeature.CompactNess= s.SurfaceArea.^1.5./s.Volume;
    globalGeoFeature.Sphericity= ((pi)^(1/3)*(6*s.Volume).^(2/3))./s.SurfaceArea;
    globalGeoFeature.Extent= s.Extent; %ratio of original volume to bounding box volume
    globalGeoFeature.Centroid=s.Centroid;
    if numel(unique(Image(:))) < 3 % it is not multiobject image and measure this feature
        [centerValue, centerLocation] = findInteriorMostPoint(Image);
        globalGeoFeature.InteriorPoint=[centerValue, centerLocation];
    else
        globalGeoFeature.InteriorPoint = [];
    end
    globalGeoFeature.AspectRatio=min(s.PrincipalAxisLength,[],2)./max(s.PrincipalAxisLength,[],2);

    %calculate roughness based on the created convexhull
    for iCell = 1: length(unique(Image(:))) - 1 % exclude the background
        s_convHull=regionprops3(s.ConvexImage{iCell},'Volume','SurfaceArea');
        s_convHull(s_convHull.Volume==0,:)=[]; %to clean up the nonrelevent output of regionprop3
        SurfaceArea(iCell,1) = s_convHull.SurfaceArea;
    end
    globalGeoFeature.Roughness=s.SurfaceArea./SurfaceArea;

    %calculate the longest length using point clouds from convex hull
    %points
    % for iCell = 1: length(unique(Image(:))) - 1 % exclude bg
    %  points3D=s.ConvexHull{iCell}; % vertex coordinate of a polygon
    %    distTemp = pdist(points3D);
    %  maxL(iCell,1) = max(distTemp(:)); minL(iCell,1) = min(distTemp(:));
    % end
    %calculate the longest length usig the surface of the image
    for iCell = 1: length(unique(Image(:))) - 1 % exclude bg
        Img =s.Image{iCell}; % vertex coordinate of a polygon
        ImgSurf = bwperim(Img,26); %  26-connectivity → 3D surface
        [Y, X, Z] = ind2sub(size(ImgSurf), find(ImgSurf));
        points3D = [Y X Z];
        distTemp = pdist(points3D);
        maxL(iCell,1) = max(distTemp(:));
        % minL(iCell,1) = min(distTemp(:)); meaningless
    end
    globalGeoFeature.NLongLength=maxL./s.Volume;
    globalGeoFeature.LongLength = maxL;
    % globalGeoFeature.NShortLength=min(distTemp(:))/s.Volume; % it is
    % meaningless globalGeoFeature.ShortLength=min(distTemp(:));


    %     %calculate the distance from the center for each face on the
    %     surface
    %     distFromCenter=distFromInteriorCenter3D(image,surface,neighbor);
    %     globalGeoFeature.VardistFromCenter=distFromCenter;
    % %
    %     %calculate the angle between normal vector and vector from center
    %     to %face for each face
    %     angleNormalVec_Centroid=angleFromInteriorCenter3D(image,surface,neighbor);
    %     globalGeoFeature.angleFace=angleNormalVec_Centroid;
    %
    %calculate the circumscribed sphericity, (Roshan),
    for iCell = 1: length(unique(Image(:))) - 1 % exclude bg
        Img = s.Image{iCell};
        [rows, cols, z] = findND(Img); %Find non-zero elements in ND-arrays
        NonZeroMatrix = [rows, cols, z];
        %find the radius and center of the minimum bounding sphere using
        %external function.
        % [circumscribed_center, circumscribed_radius] =
        % minboundsphere(NonZeroMatrix); % random search, not optmized I
        % used this package based on Ritter's algorithm
        %https://www.mathworks.com/matlabcentral/fileexchange/48725-exact-minimum-bounding-spheres-and-circles
        [circumscribed_radius, circumscribed_center] = ApproxMinBoundSphereND(NonZeroMatrix);

        %property of circumscribed Sphere
        circumscribed(iCell,1).center=circumscribed_center;
        circumscribed(iCell,1).radius=circumscribed_radius;
        circumscribed(iCell,1).volume=4/3 * pi * (circumscribed_radius.^3);
        circumscribed(iCell,1).surfaceArea = 4 * pi * (circumscribed_radius .^2);

        globalGeoFeature.CircumscribedSphere{iCell,1}=circumscribed;



        %calculate the inscribed sphericity, (Roshan),
        [inscribed_radius, inscribed_center] = findInteriorMostPoint(Img); %find the innermost point of the image; this will be
        %property of inscribed Sphere
        InscribedSphere(iCell,1).center=inscribed_center;
        InscribedSphere(iCell,1).radius=inscribed_radius;
        InscribedSphere(iCell,1).volume=4/3 * pi * (inscribed_radius.^3);
        InscribedSphere(iCell,1).surfaceArea = 4 * pi * (inscribed_radius^2);
        globalGeoFeature.InscribedSphere{iCell,1}=InscribedSphere;
    end

    globalGeoFeature.CircumscribedSurfaceRatio=globalGeoFeature.SurfaceArea./ ...
        [ circumscribed.surfaceArea]';
    %define the various sphericity
    VolumeSphericity=globalGeoFeature.Volume./[circumscribed.volume]';
    globalGeoFeature.VolumeSphericity=VolumeSphericity;

    RadiusSphericity=(globalGeoFeature.EquivDiameter/2)./[circumscribed.radius]';
    globalGeoFeature.RadiusSphericity=RadiusSphericity;

    RatioSphericity=[InscribedSphere.radius]'./[circumscribed.radius]';
    globalGeoFeature.RatioSphericity=RatioSphericity;

elseif size(Image,3) == 1 %2D image
    s = regionprops(Image,"ConvexHull","Area",'ConvexImage',...
        'ConvexArea','Centroid',"BoundingBox","EquivDiameter",'Extent','Solidity',...
        'Image', 'Perimeter','Eccentricity','SubarrayIdx');
    s(s.Area==0,:)=[];%to clean up the nonrelevent output of regionprop
    globalGeoFeature.Area=s.Area; % in pixels^2
    globalGeoFeature.Perimeter= s.Perimeter;
    globalGeoFeature.Solidity= s.Solidity;
    globalGeoFeature.Eccentricity= s.Eccentricity;
    globalGeoFeature.Extent= s.Extent;
    globalGeoFeature.Centroid=s.Centroid;
    % %calculate the longest length
    %    points2D=s.ConvexHull{end}; % vertex coordinate of a polygon
    %    distTemp=zeros(size(points2D,1)); for ii=1:size(points2D,1) %find
    %    the distance between each vertices
    %        for jj=ii:size(points2D,1)
    %            distTemp(ii,jj)=sqrt((points2D(ii,1)-points2D(jj,1))^2 ...
    %                +(points2D(ii,2)-points2D(jj,2))^2+ ...
    %                (points2D(ii,3)-points2D(jj,3))^2);
    %        end
    %    end [r c]=find(distTemp== max(distTemp(:)));


    for iCell = 1: length(unique(Image(:))) - 1 % exclude bg
        Img =s.Image{iCell}; % vertex coordinate of a polygon
        ImgSurf = bwperim(Img,8); %  26-connectivity → 3D surface
        [Y, X, Z] = ind2sub(size(ImgSurf), find(ImgSurf));
        points3D = [Y X Z];
        distTemp = pdist(points3D);
        maxL(iCell,1) = max(distTemp(:));
        % minL(iCell,1) = min(distTemp(:)); meaningless
 end
    globalGeoFeature.NLongLength=maxL./s.Area;
    globalGeoFeature.LongLength = maxL; 

   
    
for iCell = 1: length(unique(Image(:))) - 1 % exclude the background
    s_convHull=regionprops(s.ConvexImage{1},'Area','Perimeter');
    s_convHull(s_convHull.Area==0,:)=[]; %to clean up the nonrelevent output of regionprop3
    Perimeter(iCell,1) = s_convHull.Perimeter; 
    end 
    globalGeoFeature.Roughness=s.Perimeter./Perimeter;
    
end
Image=s.Image;% it needs to be the last one. 
convexImage=s.ConvexImage;

end
