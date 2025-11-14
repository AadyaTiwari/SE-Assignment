# SE-Assignment

## Objective - 

You are provided with a front-facing dashcam video (challenge-mle2.mp4) of a vehicle
driving on a highway. Your task is to detect and track all nearby vehicles and pedestrians
across the entire video, assigning consistent IDs to each object over time.

This is a multi-object tracking (MOT) challenge using monocular video. You are expected to
maintain tracking IDs across occlusions, exits, and re-entries — not just bounding boxes.


## Tools and Frameworks used 

- Python3 
- OpenCV
- Ultralytics (YOLOv8)
- NumPy
- CSV 

## Approach 
We first defined the constants like
- Input and output files
- Model path (we have used YOLOv8n here)
- Confidence threshold to reduce noise
- Target classes - list of COCO (Common Objects in Context) class IDs for pedestrians and vehicles 
- Distance to pixel ratio - Assumed as 0.2 
- Colours for the text and bounding boxes 
______________________________________________________________________

We then Initialised the required objects to process the input and achieve the objective
- Model object: YOLO model for object detection and tracking
- capture object: For reading the input video frame by frame and accessing its different attributes like 
    + frame height
    + frame width
    + frame rate and other data
- video writer object: This object is responsible for writing the annotated frames to the output file with the correct data
______________________________________________________________________

The Main Loop

- The read() method of the VideoCapture class returns a Boolean (whether a frame was read or not) and the image (BGR values of the pixels in a NumPy ndarray)
- The track() method of the YOLO class is used for Multi object Tracking (MOT)
    + It uses the loaded **YOLOv8n** model to detect object in current frame
    + It feeds the above detections into a tracking algoritm (used **bytetrack** here) that maintains the continuity of objects in the current and previos frames assigning a unique ID to each object. 
- Since we are processing one frame at a time, we get one ultralytics.engine.results.Results object from the track() function. **results[0].boxes** has all the raw data like bounding box co-ordinates, class IDs and unique IDs are available which are further accessed and stored in respective data structures 
- We then iterate through all the detected and tracked objects to unpack the co ordinates and map them as integers to ready them for input in OpenCV functions
- We then track the history of the object by adding its centre point in the deque and then calculate the length that is travelled by the object to calculate its velocity. 
- we then use the rectangle() cv2 method to outline the detected objects. 
- Generate label texts for object id and speed and add it to the label rectangles and save the data in the csv file
- we then write the annotated frame to the output file
- once the loop completes, we release/close all files 

