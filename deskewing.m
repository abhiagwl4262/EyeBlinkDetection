%% Deskewing Using RADON transform

function c = deskewing(a)

% reading and showing actual tilted image
% a = imread(filename);

% taking complement of the image
b = imcomplement(a);

% calculating radon transform of image
R = radon(b);

% finding angle of maximum value of Transform 
[r,c] = size(R);
max = 0;
ang = 0;
for i = 1 : r;
    for j = 1 : c;
        if R(i,j) > max
            max = R(i,j);
            ang = j;
        end;
    end;
end;

% deskewing
c = imrotate(b,90 - ang);
c = imcomplement(c);

%figure(2) ; imshow(c);