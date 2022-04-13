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

# define S3 bucket name
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
    segments are label: datetime_xmin_ymin_xmax_ymax.jpg
    """
    extention = '.jpg'

    s3 = boto3.client('s3', aws_access_key_id = ACCESS_KEY_ID, aws_secret_access_key = SECRET_ACCESS_KEY)

    # name photo
    photo_name = datetime.now().strftime("%d%b%Y%H%M%S")

    # post the bounding box to be labeled 
    height, width = image.shape[:2]

    for obj in objects:
        xmin, ymin, xmax, ymax = int(obj[0] * width), int(obj[1] * height), int(obj[2] * width), int(obj[3] * height)

        # crop out segmented image
        segment = image[ymin:ymax, xmin:xmax]

        # resize segment to 224, 224 !change for different vectorization steps 
        segment = cv2.resize(segment, (224, 224))

        # encode the image
        buffer = Image.fromarray(segment[::-1])
        to_post = BytesIO()
        buffer.save(to_post, format = extention)
        to_post.seek(0)

        # post the segments 
        filepath = f'segments/{photo_name}_{obj[0]:.4}_{obj[1]:.4}_{obj[2]:.4}_{obj[3]:.4}'.replace('0.', '') + extention
        s3.upload_fileobj(to_post, S3_BUCKET, filepath)

    # encode the buffer convert buffer to bytes than to ascii
    buffer = Image.fromarray(image[::-1])
    to_post = BytesIO()
    buffer.save(to_post, format = extention)
    to_post.seek(0)
    
    # post full photo
    s3.upload_fileobj(to_post, S3_BUCKET, 'images/' + photo_name + extention)

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
            # resize the image to size
            image = cv2.resize(frame, (640, 640))

            async_upload_photo.apply_async(args = [image, objects])

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

