vidReader = VideoReader('sample.mp4');
i = 1;

%%while hasFrame(vidReader)
while i < 10            
    frame = read(vidReader,i);
    imshow(frame)
    saveas(gcf,'im.png');
    i = i+1;
end;