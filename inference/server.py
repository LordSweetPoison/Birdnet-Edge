import cv2
from cv2 import exp
from infer import ObjectDetector

from flask import Flask, render_template, Response

from openvino.inference_engine import IECore

app = Flask(__name__)

model_name = 'yolov5n'
device = 'MYRIAD'

print('Setting up network for Intel NCS2')

object_detector = ObjectDetector(model_name, device)

print('Intel NCS2 succesfully initiated')

camera = cv2.VideoCapture(0)

LOGGER = print

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame

        if not success: # if the camera read was not sucessful 
            LOGGER('Camera read has failed') 
            break # end the loop is the camera has failed 

        # get the image with bounding boxes and post that online
        detections, birds_in_photo = object_detector(frame, return_detected = True)

        # encode the image then convert to bytes 
        _, buffer = cv2.imencode('.jpg', detections)
        out = buffer.tobytes()

        # yield the output 
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + out + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # run the app, debug needs to be false becuase it will try to reconnect to the ncs2
    app.run(debug=False, host="0.0.0.0")

