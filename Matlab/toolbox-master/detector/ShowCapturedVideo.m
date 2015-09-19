%% Initialization
% % initialize the video writer for saving original video
% OriVideo = VideoWriter('/Users/Alex/Desktop/HumanDetection/original.avi');
% OriVideo.FrameRate = 10;
% open(OriVideo);

% % initialize the vidoe writer for saving the detected human video
% DetVideo = VideoWriter('/Users/Alex/Desktop/HumanDetection/detected.avi');
% DetVideo.FrameRate = 10 ;
% open(DetVideo);

% load pre-trained Caltech dataset detector and camera calibration
% perameters
load('models/AcfCaltech+Detector.mat')
load('M.mat')
ShowWhichVideoResult = 'videos/original.avi';

% % get video input object
% vid = videoinput('macvideo',1);
% triggerconfig(vid, 'manual');
% start (vid)
% set(vid, 'ReturnedColorSpace', 'RGB')

% shape inserter for draw bbs on detected video
shapeInserter = vision.ShapeInserter('BorderColor','Custom', ...
'CustomBorderColor', uint8([0 255 0]), 'LineWidth', 2);

% % read video
xyloObj = VideoReader(ShowWhichVideoResult);

% creat array for saving real detected positions
real_positions = {};

% creat struct for saving each frame detections
Detections = struct(...
    'bx', {}, ...
    'by', {}, ...
    'xp', {}, ...
    'yp', {}, ...
    'ht', {}, ...
    'wd', {}, ...
    'sc', {}, ...
    'xi', {}, ...
    'yi', {}, ...
    'xw', {}, ...
    'yw', {});

%% detection and show results
figure
for i = 1:xyloObj.NumberOfFrames;
% for i = 1:200;
    % real time human detection on video
    subplot(1,2,1)
%   % capture frame from web camera
    % im = getsnapshot(vid);
%   % capture frame from PETS dataset for multi-tracking 
    % im = imread(['img/' sprintf('frame_%04d.jpg', i)]);
%   % capture frame from original video 
    im = read(xyloObj, i);
  
    % resize image for detection speed
    % im = imresize(im, 0.6);
    % write the original video into file  60% image will be loaded quicker
    % writeVideo(OriVideo,im);
    
    im = imresize(im, 0.6);
    % detect bbs in each frame
    bbs=acfDetect(im, detector); % [x y w h]
    % filter the bbs with score bigger than 50
    indice = bbs(:,5)>45;
    bbs = bbs(indice, :);
    % show result
    imshow(im);
    map = imread('map.jpg');
    % show map on 2nd subplot
    subplot(1,2,2);
    imshow(map);
    hold on;
    
    % compute human real height
    top = [bbs(:,1)+bbs(:,3)/2, bbs(:,2)];
    bot = [bbs(:,1)+bbs(:,3)/2, bbs(:,2)+bbs(:,4)];
    H = zeros(size(bbs,1), 1);
    P_bot = zeros(4,1);
    P_top = zeros(4,1);
    real_positions = zeros(size(bbs,1),2);
    range = zeros(size(bbs,1),1);
    % loop for each bb
    for j = 1 : size(bbs, 1)
       % get image position of each bounding box
       p_top = [top(j,:),1]';
       p_bot = [bot(j,:),1]';
       % compute real world bbs bottom position 
       P_bot_trans = M(:,[1,2,4]) \ p_bot;% inv(M)*p_top
       P_bot_trans = P_bot_trans / P_bot_trans(3);
       P_bot([1,2,4]) = P_bot_trans;
       P_bot(3) = 0;
       % computer real world bbs top height
       P_top_trans = M(:,[2,3,4])\(p_top-5*M(:,1));
       H(j) = P_top_trans(2)/P_top_trans(3);
       % write true human height on video
       subplot(1,2,1);
       text(bbs(j,1),bbs(j,2)-25,['People height=' num2str(H(j))],'FontSize',10,'color','w'); %[sprintf('x=%04d', P_bot(1)) sprintf('y=%04d', P_bot(2))]
       % record all detections
       real_positions(j,:) = P_bot(1:2);
%      % compute whether the bbs in ground truth range or not
        
%        judge1 = 0.5*bot(j,1)+bot(j,2)-352.5;
%        judge2 = 0.133*bot(j,1)-bot(j,2)+131;
%        if (judge1>0 && judge2>0)
%            range(j) = 1;
%        else
%            range(j) = 0;
%        end

       % draw points on 2-d plane
       if (H(j)>0.5 && H(j)<2.8)
           subplot(1,2,2);
           p_map = [224-P_bot(2)*82/1.4, 356-P_bot(1)*82/1.4];
           plot(p_map(1),p_map(2),'.','MarkerSize',30);
       end
    end
    subplot(1,2,1);
    % draw the passed bbs
    indice = H>0.5 & H<2.8;
    bbs = bbs(indice, :);
    real_positions = real_positions(indice, :);
%   % draw the bbs in ground truth area
%     indice = find(range ==1);
%     bbs = bbs(indice, :);
    bbApply('draw',bbs);
    pause(.05);
  % write detected video
     if ~isempty(bbs)
         rectangles = int32(bbs(:,1:4));
         % rectangles([1,2,3,4],:) = rectangles([2,1,4,3],:);
         DetIm = step(shapeInserter, im, rectangles); % [row column height width]
     else
         DetIm = im;
     end
     % writeVideo(DetVideo,DetIm);
     % Create a new detection
     numOfBoxes = size(bbs, 1);
     bx = zeros(1, numOfBoxes);
     by = zeros(1, numOfBoxes);
     xp = zeros(1, numOfBoxes);
     yp = zeros(1, numOfBoxes);
     ht = zeros(1, numOfBoxes);
     wd = zeros(1, numOfBoxes);
     sc = zeros(1, numOfBoxes);
     xi = zeros(1, numOfBoxes);
     yi = zeros(1, numOfBoxes);
     xw = zeros(1, numOfBoxes);
     yw = zeros(1, numOfBoxes);
     
     for j = 1: numOfBoxes
        bx(j) = bbs(j, 1);
        by(j) = bbs(j, 2);
        xp(j) = real_positions(1)*1000;
        yp(j) = real_positions(2)*1000;
        ht(j) = bbs(j, 4);
        wd(j) = bbs(j, 3);
        sc(j) = bbs(j, 5);
        xi(j) = bbs(j, 1)+bbs(j, 3)/2;
        yi(j) = bbs(j, 2)+bbs(j, 4);
        xw(j) = xp(j);
        yw(j) = yp(j);
         
     end
     
     newDetection = struct(...
    'bx', bx, ...  % bounding box left top x (pixel)
    'by', by, ...  % bounding box left top y (pixel)
    'xp', xp, ...  % bottom center real world position x (mm)
    'yp', yp, ...  % bottom center real world position y (mm)
    'ht', ht, ...  % bounding box height (pixel)
    'wd', wd, ...  % bounding box width (pixel)
    'sc', sc, ...  % score of each bounding box 
    'xi', xi, ...  % bottom center image position x (pixel)
    'yi', yi, ...  % bottom center image position y (pixel)
    'xw', xw, ...  % same as xp
    'yw', yw);     % same as yp
    
    % add new detection to Detections 
    Detections(end+1) = newDetection;
    
end
% % close the opened video writer
% close(OriVideo);
% close(DetVideo);
% % close camera input object
% stop(vid)
% delete(imaqfind)


% I = imread('/Users/Alex/Desktop/HumanDetection/one.jpg');
% 
% bbs=acfDetect(I, detector);
% figure(1); im(I); bbApply('draw',bbs); pause(.1);