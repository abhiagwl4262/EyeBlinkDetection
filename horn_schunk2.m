%% Here convolution is done as seperable convolution

function horn_schunk2(filename)
obj = VideoReader(filename);
frames = obj.Numberofframes;
h = fspecial('average', [3,3]);

hor_flow = zeros(frames,1);
ver_flow = zeros(frames,1);
flow = zeros(frames,1);
hor_diff = zeros(frames,1);
ver_diff = zeros(frames,1);
overall_flow = zeros(frames,1);

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
    kernal1 = [-1 0 1; -2 0 2; -1 0 1];
    kernal2 = [0 0 0; 0 1 0; 0 0 0];
    
    kernal1_h = [-1 0 1];
    kernal1_v = [1; 2; 1];
    Ix = conv2(frame1,kernal1_h,'same');
    Ix = conv2(frame1,kernal1_v,'same');
    
    Iy = conv2(frame1,kernal1_v','same');
    Iy = conv2(frame1,kernal1_h','same');
    
    It = frame1-frame2;
    %Ix = conv2(frame1,kernal1,'same');
    %Iy = conv2(frame1,kernal1','same');
    %It = conv2(frame1,kernal2,'same') + conv2(frame2,-kernal2,'same');

    %% Set initial value of u and v to zero
    u = 0;
    v = 0;
    alpha  = 10;
    kernal_avg = [0 1 0; 1 0 1; 0 1 0]; % Average kernel
    
    %% Minimizing Iterations
    for i=1:100
        %Computing local averages of the vectors
        uAvg=conv2(u,kernal_avg,'same');
        vAvg=conv2(v,kernal_avg,'same');
        
        % Updating the flow vector
        % where alpha is the smoothing weight
        u = uAvg - (Ix.*( (Ix.*uAvg)+(Iy.*vAvg)+It) )./(alpha^2+Ix.^2+Iy.^2);
        v = vAvg - (Iy.*( (Ix.*uAvg)+(Iy.*vAvg)+It) )./(alpha^2+Ix.^2+Iy.^2);
    end

    m = sumsqr(u);
    hor_flow(num,1) = m;
    n = sumsqr(v);
    ver_flow(num,1) = n;
    value = [m n];
    flow(num,1) = sumsqr(value);
end
toc;
[rows, ~] = size(flow);
for i = 2: rows
    hor_diff(i,1) = hor_flow(i,1) - hor_flow(i-1,1); %difference of horizental flow in adjacent images
    ver_diff(i,1) = ver_flow(i,1) - ver_flow(i-1,1);
    overall_flow(i,1) = flow(i,1) - flow(i-1,1); %difference of image vector flow in adjacent images
end

hor_diff  = hor_diff/max(hor_diff); 
final_hor_flow = hor_diff > 0.5; %detecting blink of the eye

ver_diff  = ver_diff/max(ver_diff); 
final_ver_flow = ver_diff > 0.5; %detecting blink of the eye

overall_flow = overall_flow / max(overall_flow);
flow_value = overall_flow > 0.5;

save_weight('horn_schunk2_final_ver_flow_100.txt', final_ver_flow);
save_weight('horn_schunk2_final_hor_flow_100.txt', final_hor_flow);
save_weight('horn_schunk2_overall_flow_value.txt', flow_value);

toc;