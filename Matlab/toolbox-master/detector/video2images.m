


xyloObj = VideoReader('/Users/Alex/Desktop/HumanDetection/test/o4_wifi.avi');
for i  = 1: xyloObj.NumberOfFrames;
    im = read(xyloObj, i);
    im = imresize(im, 0.6);
    imwrite(im, ['o4_wifi/' sprintf('frame_%04d.jpg', i)]);
end