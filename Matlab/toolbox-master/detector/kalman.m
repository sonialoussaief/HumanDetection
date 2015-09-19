

function kalman()

% Create system objects used for reading video, detecting moving objects,
% and displaying the results.

% load pre-trained Caltech dataset detector and camera calibration
% perameters
detector = load('/Users/Alex/Desktop/HumanDetection/toolbox-master/detector/models/AcfCaltech+Detector.mat');
M = load('M.mat');
detector = detector.detector;
M = M.M;
map = imread('map.jpg');
TrackWhichVideo = 'videos/o9_wifi.avi';

% % initialize the video writer for saving original video
% OriVideo = VideoWriter('/Users/Alex/Desktop/HumanDetection/original.avi');
% OriVideo.FrameRate = 10;
% open(OriVideo);

% initialize the vidoe writer for saving the detected human video
DetVideo = VideoWriter('videos/detection&tracking.avi');
DetVideo.FrameRate = 10 ;
open(DetVideo);

obj = setupSystemObjects(); % ------ok

tracks = initializeTracks(); % Create an empty array of tracks. ----ok

nextId = 1; % ID of the next track -----ok

% foot position radius 
r = 1;
% video length
videoTime = 74;

% % get video input object
% vid = videoinput('macvideo',1);
% triggerconfig(vid, 'manual');
% start (vid)
% set(vid, 'ReturnedColorSpace', 'RGB')

% matrixes to save results
X = zeros(obj.reader.NumberOfFrames, 2);
Y = zeros(obj.reader.NumberOfFrames, 2);

frameNums = zeros(1, obj.reader.NumberOfFrames);
% Detect people, and track them across video frames.
lostInds = [];
for j = 1:obj.reader.NumberOfFrames; % ------ok
% for j = 1:200;
    frame = read(obj.reader, j); % -------ok
    % frame = getsnapshot(vid);
    % frame = flip(frame,2);
    frame = imresize(frame, 0.6);
    
    if isempty(lostInds)
         validTracks = tracks;
    else
         validTracks = tracks(~lostInds);
    end
    % writeVideo(OriVideo,frame);
    % detection return the location of tracking point and the correspnding
    % bounding box
    [locations, R_locations, foot ,bboxes, mask] = detectObjects(frame, detector, M); % [x,y]; [x,y,w,h] ------ok
    predictNewLocationsOfTracks(); % ------ ok
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment(); % ------ ok
    
    updateAssignedTracks();  % ----- ok
    updateUnassignedTracks(); % ------ ok
    lostInds = LostTracks(); % ----- ok
    validTracks = tracks(~lostInds);
    createNewTracks(); % ----- ok
    % display tracking results on both video and 2-d plane
    displayTrackingResults(); % ---- ok
    % [Tang, X, Y] = writeResult(Tang, X, Y);

    frameNums(j) = j;
end
totalVisibleCounts = [tracks(:).totalVisibleCount];
validTracks = tracks(totalVisibleCounts>30);
for j = 1: size(validTracks, 2);
    id = validTracks(j).id;
    idc = validTracks(j).realPosition(:,3)==1;
    idc_predicted = find(validTracks(j).realPosition(:,4)==1);
    officePosition = zeros(obj.reader.NumberOfFrames, 3);
    for f = 1: obj.reader.NumberOfFrames;
        officePosition(f, 1) = f*videoTime/obj.reader.NumberOfFrames;
    end
    InterPosition = validTracks(j).realPosition(idc, [2,1]); 
    InterPosition(:, 1) = 4.2-InterPosition(:, 1);
    InterPosition(:, 2) = 2.4+InterPosition(:, 2);
    officePosition(idc, 2:3) = InterPosition;
    officePosition(idc_predicted, 2:3) = zeros(size(idc_predicted, 1),2);
    csvwrite(['csv/' obj.reader.Name sprintf('_trackerID_%d.csv', id)], officePosition);
end


% csvwrite('tracking.csv',Tang);
save('work.mat');
% close(DetVideo);
% close(OriVideo);

% stop(vid)
% delete(imaqfind)
%% write the result into specific form
function [Tang, X, Y] = writeResult(Tang, X, Y)
    Tang(j, 1) = j/obj.reader.NumberOfFrames*129;
    n = size(tracks, 2);
    if n == 1
        Tang(j, 2) = 4.2-tracks(1).realPosition(2);
        Tang(j, 3) = 2.4+tracks(1).realPosition(1);
        X(j, 1) = tracks(1).realPosition(1);
        Y(j, 1) = tracks(1).realPosition(2);
    end
    if n >1
        Tang(j, 2) = 4.2-tracks(1).realPosition(2);
        Tang(j, 3) = 2.4+tracks(1).realPosition(1);
        Tang(j, 4) = 4.2-tracks(2).realPosition(2);
        Tang(j, 5) = 2.4+tracks(2).realPosition(1);
        X(j, 1) = tracks(1).realPosition(1);
        Y(j, 1) = tracks(1).realPosition(2);
        X(j, 2) = tracks(2).realPosition(1);
        Y(j, 2) = tracks(2).realPosition(2);
    end
    
end

%% Create System Objects
% Create System objects used for reading the video frames

    function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.
        
        % Create a video reader.
        obj.reader = VideoReader(TrackWhichVideo);
        
        % Create video players to display the video
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
      
        obj.mapPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
        
        % Create system objects for foreground detection and blob analysis
        
        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background.2
        
        obj.gaussian = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
        
        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis system object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, location, and the bounding box.
        
%         obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
%             'AreaOutputPort', true, 'locationOutputPort', true, ...
%             'MinimumBlobArea', 400);
    end

%% Initialize Tracks
    % 1*2 struct with 8 fields
    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {}, ...
            'realPosition', {}, ...
            'footPosition', {});
        validTracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {}, ...
            'realPosition', {}, ...
            'footPosition', {});
    end


%% Detect Objects

% Use ACF to detect human

    function [locations,R_locations, fPositions, bboxes,mask] = detectObjects(frame, detector, M)
        
        
        % Detect human and return the [bounding box bottom center position] 
        % [bbox bot center real position] and [bounding boxes]
        % detect bbs in each frame
        bbs=acfDetect(frame, detector); % [x y w h]
        % filter the bbs with score bigger than 50
        indice = bbs(:,5)>45;
        bbs = bbs(indice, :);
        
        % use gaussian mixture model to get the foot more precise position 
        
        mask = obj.gaussian.step(frame);
        % Apply morphological operations to remove noise and fill in holes.
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15])); 
        mask = imfill(mask, 'holes');
        fPositions = zeros(size(bbs,1),3);
        
        % compute human real height
        % if there are k bounding boxes; top and bot are both k*2 matrixes
        top = [bbs(:,1)+bbs(:,3)/2, bbs(:,2)];  
        bot = [bbs(:,1)+bbs(:,3)/2, bbs(:,2)+bbs(:,4)];
        H = zeros(size(bbs,1), 1); % H is k*1 for storing each box real height
        P_bot = zeros(4,1);
        % detections in each frame
        detections = zeros(size(bbs,1),4);
        % loop for each bb
        for b = 1 : size(bbs, 1)
           % get image position of each bounding box
           p_top = [top(b,:),1]';
           p_bot = [bot(b,:),1]';
           % compute each bbs real world bottom position 
           P_bot_trans = M(:,[1,2,4]) \ p_bot;% inv(M)*p_top
           P_bot_trans = P_bot_trans / P_bot_trans(3);
           P_bot([1,2,4]) = P_bot_trans;
           P_bot(3) = 0;
           % computer real world bbs top height
           P_top_trans = M(:,[2,3,4])\(p_top-5*M(:,1));
           H(b) = P_top_trans(2)/P_top_trans(3);
           
           % record all detections' real bottom locations
           detections(b,:) = [P_bot(1), P_bot(2), r, 0];
           % find out the lowest mask position in each bounding box as foot
           % position
%            window = mask(bbs(b,2):(bbs(b,2)+bbs(b,4)),bbs(b,1):(bbs(b,1)+bbs(b,3)));
%            [row, ~] = find(window==1);
%            if max(row)>size(window,1)*7/8
%                 f_row = max(row)+bbs(b,2);
%            else
%                f_row = bbs(b,2)+bbs(b,4)*0.9;
%            end
%            f_col = bbs(b,1)+bbs(b,3)/2;
           fPositions(b,:) = [(bbs(b,1)+bbs(b,3)/2), (bbs(b,2)+bbs(b,4)), r];
        end
        % convert float results into integer 
        indice = H>0.5 & H<2.8;
        locations = int32(bot(indice,:));  % in each frame, more than 1 location
        R_locations = detections(indice,:); % b*3 matrix last column is radius of 2-d map dot 
        bboxes = int32(bbs(indice, 1:4));   
        fPositions = int32(fPositions(indice,:));
    end

%% Predict New Locations of Existing Tracks
% Use the Kalman filter to predict the location of each track in the
% current frame, and update its bounding box accordingly.

    function predictNewLocationsOfTracks()
        for i = 1:length(validTracks)
            bbox = validTracks(i).bbox;
            footP = validTracks(i).footPosition;
            % Predict the current location of each track.
            predictedlocation = predict(validTracks(i).kalmanFilter);
            
            % Shift the bounding box so that it is at the predicted location.
            topleft = [predictedlocation(1)-bbox(3)/2, predictedlocation(2)-bbox(4)]; 
            validTracks(i).bbox = [topleft, bbox(3:4)];
            
    %%%%%%%%% update foot position   
            
            validTracks(i).footPosition = [footP(1)+(predictedlocation(1)-(bbox(1)+bbox(3)/2)), ...
                footP(2)+(predictedlocation(2)-(bbox(2)+bbox(4))), r];
    %%%%%%%%% update real position 
            % get image position of each bounding box
           
           p_bot = [(bbox(1)+bbox(3)/2),(bbox(2)+bbox(4)),1]';
           % compute each bbs real world bottom position 
           P_bot_trans = M(:,[1,2,4]) \ double(p_bot); % inv(M(:,[1,2,4]))*p_top
           P_bot_trans = P_bot_trans / P_bot_trans(3);
           P_bot([1,2,4]) = P_bot_trans;
           P_bot(3) = 0;
           validTracks(i).realPosition(j,:) = [P_bot(1), P_bot(2), r, validTracks(i).consecutiveInvisibleCount>0];  
        end
        if isempty(lostInds)
            tracks = validTracks;
        else
            tracks(~lostInds) = validTracks;
        end
    end

%% Assign Detections to Tracks

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
        nTracks = length(validTracks);
        nDetections = size(locations, 1);
        
        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(validTracks(i).kalmanFilter, locations);
        end
        
        % Solve the assignment problem.
        % distance for not assign detection to tracker
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
    end

%% Update Assigned Tracks
% The |updateAssignedTracks| function updates each assigned track with the
% corresponding detection. It calls the |correct| method of
% |vision.KalmanFilter| to correct the location estimate. Next, it stores
% the new bounding box, and increases the age of the track and the total
% visible count by 1. Finally, the function sets the invisible count to 0. 

    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            location = locations(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            footP = foot(detectionIdx,:);
            realP = R_locations(detectionIdx,:);
            
            % Correct the estimate of the object's location
            % using the new detection.
            correct(validTracks(trackIdx).kalmanFilter, location);
            
            % Replace predicted bounding box with detected
            % bounding box.
            validTracks(trackIdx).bbox = bbox;
            
            % Update track's age.
            validTracks(trackIdx).age = validTracks(trackIdx).age + 1;
            
            % Update visibility.
            validTracks(trackIdx).totalVisibleCount = ...
                validTracks(trackIdx).totalVisibleCount + 1;
            validTracks(trackIdx).consecutiveInvisibleCount = 0;
            
            % Update foot position
            validTracks(trackIdx).footPosition = footP;
            
            % Update real position
            validTracks(trackIdx).realPosition(j,:) = realP;
        end
        if isempty(lostInds)
            tracks = validTracks;
        else
            tracks(~lostInds) = validTracks;
        end
    end

%% Update Unassigned Tracks
% Mark each unassigned track as invisible, and increase its age by 1.

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            validTracks(ind).age = validTracks(ind).age + 1;
            validTracks(ind).consecutiveInvisibleCount = ...
                validTracks(ind).consecutiveInvisibleCount + 1;
        end
        if isempty(lostInds)
            tracks = validTracks;
        else
            tracks(~lostInds) = validTracks;
        end
    end

%% Delete Lost Tracks
% The |deleteLostTracks| function deletes tracks that have been invisible
% for too many consecutive frames. It also deletes recently created tracks
% that have been invisible for too many frames overall. 

    function lostInds = LostTracks()
        if isempty(tracks)
            lostInds = [];
            return
        end
        
        invisibleForTooLong = 20;
        ageThreshold = 8;
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        % Delete lost tracks.
        % tracks = tracks(~lostInds);
    end

%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned
% detection is a start of a new track. In practice, you can use other cues
% to eliminate noisy detections, such as size, location, or appearance.

    function createNewTracks()
        locations = locations(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        fPs = foot(unassignedDetections, :);
        rPs = R_locations(unassignedDetections, :);
        
        for i = 1:size(locations, 1)
            real_position_matrix = zeros(obj.reader.NumberOfFrames, 4);
            location = locations(i,:);
            bbox = bboxes(i, :);
            fP = fPs(i,:);
            rP = rPs(i,:);
            real_position_matrix(j, :) = rP; 
            
            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                location, [200, 50], [100, 25], 100);
            
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0, ...
                'realPosition', real_position_matrix, ...
                'footPosition', fP);
            
            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;
            
            % Increment the next id.
            nextId = nextId + 1;
        end
    end

%% Display Tracking Results
% The |displayTrackingResults| function draws a bounding box and label ID 
% for each track on the video frame and the foreground mask. It then 
% displays the frame and the mask in their respective video players. 

    function displayTrackingResults()
        
        validTracks = tracks(~lostInds);
        % Convert the frame and the mask to uint8 RGB.
       
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        minVisibleCount = 3; % 8
        if ~isempty(validTracks)
              
            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than 
            % a minimum number of frames.
            reliableTrackInds = ...
                [validTracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = validTracks(reliableTrackInds);
            
            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                
                bboxes = cat(1, reliableTracks.bbox);
                predicted = cat(1, reliableTracks.consecutiveInvisibleCount);
                predictedIndc = predicted>0;
                circle = ones(size(reliableTracks, 2), 3);
                for t = 1: size(reliableTracks, 2)
                    circle(t, :) = reliableTracks(t).realPosition(j,1:3);
                end
                circle_map = ones(size(circle));
                circle_map(:,1) =  224-circle(:, 2)*82/1.4;
                circle_map(:,2) =  356-circle(:, 1)*82/1.4;
                % Get ids.
                ids = int32([reliableTracks(:).id]);
                
                % Create labels for objects indicating the ones for 
                % which we display the predicted rather than the actual 
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {'predicted'};
                labels = strcat(labels, isPredicted);
                
                real_bboxes = bboxes;
                length = size(find(predictedIndc>0), 1);
                box_0 = zeros(length, 4);
                real_bboxes(predictedIndc,:) = box_0;
                
                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    real_bboxes, labels);
                % Draw the objects on the mask.
                circle_map = int32(circle_map);
                map2 = insertObjectAnnotation(map, 'circle', ...
                    circle_map, labels);
            end
        end
        
        % save the detected video
        writeVideo(DetVideo,frame);
        % Display the mask and the frame.       
        obj.videoPlayer.step(frame);
        if (~isempty(validTracks)) && (~isempty(reliableTracks))
            obj.mapPlayer.step(map2);
        else 
            obj.mapPlayer.step(map);
        end
    end


end

