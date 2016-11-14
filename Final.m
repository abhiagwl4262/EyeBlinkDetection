clear; close all;

% Setting Key Thresold Values
MIN_PERIMETER = 500;
MIN_AREA = 100;
MAX_AREA = 1000;

% Reading Input Image
A = imread('/Users/ykg2910/Documents/MATLAB/ANPR/dataset/cars_markus/img_1.jpg'); 
% Preprocessing
image = rgb2gray(A);
image = imresize(image,[720 960]);
figure(1), imshow(image);

%% Applying Edging on Image
Edge = edge(image);
se=strel('disk',1);
figure(2), imshow(Edge);

%% Processing Boundries Based on Perimeter of Contours
[B,L] = bwboundaries(Edge,'noholes');
perimeter = zeros(length(B),1);

% loop over the boundaries
for k = 1:length(B)
  
  % Boundary coordinates corresponding to label 'k'
  boundary = B{k};   
  % compute a simple estimate of the object's perimeter
  delta_sq = diff(boundary).^2;    
  perimeter(k) = sum(sqrt(sum(delta_sq,2)));
  
  if perimeter(k) >= MIN_PERIMETER
      selection = bwselect(Edge,boundary(1,2),boundary(1,1));
      selection = imcomplement(selection);
      L = L.*selection;      
  end
  
end

Edge = im2bw(L);
figure(3), imshow(Edge);

%% Processing Contours based on AREA

Mask = imfill(Edge,'holes');
Mask = bwareaopen(Mask,MIN_AREA);  %remove areas less than a particular no. of pixels
CC = bwconncomp(Mask, 8); %Determine the connected components:
S = regionprops(CC, 'Area'); %Compute the area of each component:

%Remove large objects:
L = labelmatrix(CC);
Mask = ismember(L, find([S.Area] <= MAX_AREA));        
Mask = imclearborder(Mask,8);

Mask = imopen(Mask,se);
figure(4), imshow(Mask);

%% Segmenting Individual Characters from License Plate Region

C = find_Coordinatesnew(Mask);
crop = image(C(2,1)-20:C(2,2)+20,C(1,1)-20:C(1,2)+20);
crop = adaptivethreshold(crop,10,0.1,0);

final = deskewing(crop);
figure(5), imshow(crop);

[characters, final] = segment1(final);

%% Displaying Individual Character
for i=1:size(characters)
    imshow(characters{i});
    i = i+1;
    disp('press any key to continue')
    pause;
end
    
figure(7), imshow(final);