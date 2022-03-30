import cv2
from datetime import datetime 
import requests
import base64

from infer import ObjectDetector

from flask import Flask, render_template, Response, jsonify

from celery import Celery

from config import ROBOFLOW_PRIVATE

app = Flask(__name__)

# define celery redis location 
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# define celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# set up nsc2 and camera only if name is main
if __name__ == "__main__":
    model_name = 'yolov5n'
    device = 'MYRIAD'

    print('Setting up network for Intel NCS2')

    object_detector = ObjectDetector(model_name, device, num_classes = 1, conf_threshold = .2)

    print('Intel NCS2 succesfully initiated')

    camera = cv2.VideoCapture(0)

LOGGER = print

@celery.task
def async_upload_photo(to_post):
    # define daatset
    dataset = 'birdcamid'

    # name photo
    photo_name = datetime.now().strftime("%d%b%Y%H%M%S")

    # create upload url
    upload_url = "".join([
        f"https://api.roboflow.com/dataset/{dataset}/upload",
        f"?api_key={ROBOFLOW_PRIVATE}",
        f"&name={photo_name}.jpg",
        "&split=train"
    ])
    # post request to upload photo
    r = requests.post(upload_url, data = to_post, headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    })


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

        # if birds are in the photo send to celery to upload
        if birds_in_photo:
            # resize the image to size
            image = cv2.resize(frame, (640, 640))

            # encode the buffer
            _, buffer = cv2.imencode('.jpg', image)

            # convert buffer to bytes than to ascii
            to_post = base64.b64encode(buffer).decode('ascii')

            # async send photo to roboflow
            async_upload_photo.apply_async(args = [to_post])

        # yield the output 
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + out + b'\r\n')
                

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
    
@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # run the app, debug needs to be false becuase it will try to reconnect to the ncs2 if debug is true
    app.run(debug=False, host="0.0.0.0")

