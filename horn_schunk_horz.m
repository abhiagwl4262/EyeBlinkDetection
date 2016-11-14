%% Here convolution is done as seperable convolution

function horn_schunk2(filename)
obj = VideoReader(filename);
frames = obj.Numberofframes;
h = fspecial('average', [3,3]);

hor_flow = zeros(frames,1);
flow = zeros(frames,1);
hor_diff = zeros(frames,1);

tic; % Initialize the timer to calculate the time consumed.

%% CAPTURING 2 FRAMES FROM A VIDEO
for num = 2:frames
    frame1 = read(obj,num);
    frame1 = imfilter(frame1,h);
    frame1 = rgb2gray(frame1);
    frame1 = im2double(frame1);
    
    frame2 = read(obj,num-1);
    frame2 = imfilter(frame2,h);
    frame2 = rgb2gray(frame2);
    frame2 = im2double(frame2);
    %% METHOD 2
  %  kernal1 = [-1 0 1; -2 0 2; -1 0 1];
  %  kernal2 = [0 0 0; 0 1 0; 0 0 0];
    
    kernal1_h = [-1 0 1];
    kernal1_v = [1; 2; 1];
    Ix = conv2(frame1,kernal1_h,'same');
    Ix = conv2(frame1,kernal1_v,'same');
    It = frame1-frame2;

    %Ix = conv2(frame1,kernal1,'same');
    %It = conv2(frame1,kernal2,'same') + conv2(frame2,-kernal2,'same');

    %% Set initial value of u and v to zero
    u = 0;
    alpha  = 10;
    kernal_avg = [0 1 0; 1 0 1; 0 1 0]; % Average kernel
    
    %% Minimizing Iterations
    for i=1:100
        %Computing local averages of the vectors
        uAvg = conv2(u,kernal_avg,'same');
        u = uAvg - (Ix.*( (Ix.*uAvg)+It) )./(alpha^2+Ix.^2);
    end

    m = sumsqr(u);
    hor_flow(num,1) = m;
end

[rows, ~] = size(flow);
for i = 2:rows
    hor_diff(i,1) = hor_flow(i,1) - hor_flow(i-1,1); %difference of horizental flow in adjacent images
end

hor_diff  = hor_diff/max(hor_diff); 
final_hor_flow = hor_diff > 0.4; %detecting blink of the eye

save_weight('horn_schunk_horz.txt', final_hor_flow);
toc;