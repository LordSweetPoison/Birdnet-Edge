import cv2
from datetime import datetime 
import boto3
from io import BytesIO
from PIL import Image
import numpy as np
import asyncio 

from infer import ObjectDetector

from config import ACCESS_KEY_ID, SECRET_ACCESS_KEY

S3_BUCKET = 'birdnet-edge-brad'

# define globals for detection
model_name = '../models/best-int8_edgetpu.tflite'
device = 'TPU'

print('Setting up Coral TPU')

object_detector = ObjectDetector(model_name, device,  num_classes = 1, conf_threshold = .2)

print('Coral TPU Successfully Initiated')

camera = cv2.VideoCapture(0)

async def async_upload_photo(image, objects):
    """upload image and segments to s3
    args: 
        image: image to be uploaded
        objects: object to be segmented 
    segments are labeled: datetime_xmin_ymin_xmax_ymax.jpg
    """
    S3 = boto3.client('s3', aws_access_key_id = ACCESS_KEY_ID, aws_secret_access_key = SECRET_ACCESS_KEY)

    extention = '.jpeg'

    # name photo
    photo_name = datetime.now().strftime("%d%b%Y%H%M%S")

    # post the bounding box to be labeled 
    height, width = image.shape[:2]

    for obj in objects:
        xmin, ymin, xmax, ymax = int(obj[0] * width), int(obj[1] * height), int(obj[2] * width), int(obj[3] * height)

        # crop out segmented image
        segment = image[ymin:ymax, xmin:xmax]

        # encode the image into a buffer
        segment = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
        buffer = Image.fromarray(segment)
        to_post = BytesIO()
        buffer.save(to_post, format = 'JPEG')
        to_post.seek(0)

        # post the segments 
        filepath = f'segments/{photo_name}_{obj[0]:.4}_{obj[1]:.4}_{obj[2]:.4}_{obj[3]:.4}'.replace('0.', '') + extention
        S3.upload_fileobj(to_post, S3_BUCKET, filepath)

    # encode the image into a buffer
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    buffer = Image.fromarray(image)
    to_post = BytesIO()
    buffer.save(to_post, format = 'JPEG')
    to_post.seek(0)
    
    # post full photo
    filepath =  'images/' + photo_name + extention
    S3.upload_fileobj(to_post, S3_BUCKET, filepath)

    return None

async def run():
    # create list of tasks
    tasks = []
    while True:
        success, frame = camera.read()  # read the camera frame

        if not success: # if the camera read was not sucessful 
            print('Camera read has failed') 
            break # end the loop is the camera has failed 

        # get the image with bounding boxes and post that online
        detections, objects = object_detector(frame, return_boxes = True)

        # if the list of birds (objects) is not empty, upload the photo 
        if objects.size > 0:
            tasks.append(asyncio.create_task(async_upload_photo(frame, objects)))

    for task in tasks:
        await task


if __name__ == '__main__':
    asyncio.run(run())