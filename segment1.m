function [A, newImg] = segment1(image)

A = cell(5,1);
%Edge = edge(image);
image = imcomplement(image);
[B,L] = bwboundaries(image,'noholes');

%Compute the area of each component:
stats = regionprops(L,'Area','BoundingBox'); %contains only one structure stats.Area

%% loop over the boundaries

%perimeter = zeros(length(B),1);    %length(B) is no. of diff elements
i = 1;
se = strel('disk',1);
for k = 1:length(B)
  % obtain (X,Y) boundary coordinates corresponding to label 'k'
  boundary = B{k};

  % compute a simple estimate of the object's perimeter
  delta_sq = diff(boundary).^2;    
  perimeter = sum(sqrt(sum(delta_sq,2)));
  
  area = stats(k).Area;
  metric = area/perimeter;
  selection = bwselect(image,boundary(1,2),boundary(1,1));  %select part with given specs
  
  if(metric<= 1.5 || area<= 100 || area>=1500)                    
      selection = imcomplement(selection);                                  
      L = L.*selection;                      %return matrix with that part removed
  else
      selection = imdilate(selection,se);
      C = stats(k).BoundingBox;   %returns [x y w h]
      A{i,1} = selection(C(1,2):C(1,2)+C(1,4),C(1,1):C(1,1)+C(1,3));                      %(w:x,h:y)
      i = i+1;
  end
end

newImg = L>0;
newImg = imdilate(newImg,se);
newImg = imcomplement(newImg);