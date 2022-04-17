import cv2
from datetime import datetime 
import requests
import base64
import boto3
from io import BytesIO
from PIL import Image

from infer import ObjectDetector

from flask import Flask, render_template, Response, jsonify

from celery import Celery

from config import ACCESS_KEY_ID, SECRET_ACCESS_KEY

# define S3 bucket name and s3 session
S3_BUCKET = 'birdnet-edge-brad'


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
def async_upload_photo(image, objects):
    """upload image and segments to s3
    args: 
        image: image to be uploaded
        objects: object to be segmented 
    segments are labeled: datetime_xmin_ymin_xmax_ymax.jpg
    """
    S3 = boto3.client('s3', aws_access_key_id = ACCESS_KEY_ID, aws_secret_access_key = SECRET_ACCESS_KEY)

    extention = '.jpg'

    # name photo
    photo_name = datetime.now().strftime("%d%b%Y%H%M%S")

    # post the bounding box to be labeled 
    height, width = image.shape[:2]

    for obj in objects:
        xmin, ymin, xmax, ymax = int(obj[0] * width), int(obj[1] * height), int(obj[2] * width), int(obj[3] * height)

        # crop out segmented image
        segment = image[ymin:ymax, xmin:xmax]

        # encode the image into a buffer
        buffer = Image.fromarray(segment[::-1])
        to_post = BytesIO()
        buffer.save(to_post, format = extention)
        to_post.seek(0)

        # post the segments 
        filepath = f'segments/{photo_name}_{obj[0]:.4}_{obj[1]:.4}_{obj[2]:.4}_{obj[3]:.4}'.replace('0.', '') + extention
        S3.upload_fileobj(to_post, S3_BUCKET, filepath)

    # encode the image into a buffer
    buffer = Image.fromarray(image[::-1])
    to_post = BytesIO()
    buffer.save(to_post, format = extention)
    to_post.seek(0)
    
    # post full photo
    filepath =  'images/' + photo_name + extention
    S3.upload_fileobj(to_post, S3_BUCKET, filepath)

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame

        if not success: # if the camera read was not sucessful 
            LOGGER('Camera read has failed') 
            break # end the loop is the camera has failed 

        # get the image with bounding boxes and post that online
        detections, objects = object_detector(frame, return_detected = True)

        # encode the image then convert to bytes 
        _, buffer = cv2.imencode('.jpg', detections)
        out = buffer.tobytes()

        # if the list of birds (objects) is not empty, upload the photo 
        if objects:
            async_upload_photo.delay(frame, objects)

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

