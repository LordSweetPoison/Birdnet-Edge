import cv2
from infer import ObjectDetector

S3_BUCKET = 'birdnet-edge-brad'

from upload import async_upload_photo

def run():
    object_detector = ObjectDetector('../models/best-int8_edgetpu.tflite', device = 'TPU', conf_threshold = .4, num_classes = 1, img_size = 448)
    
    camera = cv2.VideoCapture('/dev/video1')

    while True:
        success, frame = camera.read()  # read the camera frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not success: # if the camera read was not sucessful 
            print('Camera read has failed') 
            break # end the loop is the camera has failed 

        # get the image with bounding boxes and post that online
        detections, objects = object_detector(frame, return_boxes = True)

        # if the list of birds (objects) is not empty, upload the photo 
        if objects.size > 0:
            async_upload_photo.apply_async((frame.tolist(), objects.tolist()))

if __name__ == '__main__':
    run()