# Please read this before using the 'kalman.m' function to track objects

1) The tool-box master detector part is necessary.

2) There are four important m files: 

     *--- 'calibration.m': calibrate the camera, key points must be
          pointed out from bottom right, counterclockwise, base to top.
     *--- 'CaptureVideo.m': capture video from web camera in real time, and
          the captured video is saved in the file named 'videos' as
          'original.avi', and the first frame of the video is saved as 'cali.jpg'
          for camera calibration.
     *--- 'ShowCapturedVideo.m': show the detection result of any specific
          video, and the input video's path is set by the parameter
          'ShowWhichVideoResult'.
     *--- 'kalman.m': show the tracking result of any specific video, and 
          the input video's path is set by the parameter 'TrackWhichVideo'
          
 2) show/not show bounding box score is determined by the function 'bbApply.m' 
    in detector file at line 300.
    
 3) When calibrate the camera, just run the 'calibration.m' file and it
    will use the lastest 'cali.jpg' as input image.

 4) Previous tracking results are in the CSV folder
 
 5) Video results are in the VIDEO folder.
