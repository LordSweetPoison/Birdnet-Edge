import requests
import base64
import io
import cv2
from datetime import datetime

from config import ROBOFLOW_PUBLIC, ROBOFLOW_PRIVATE

def upload_cv2_image(image, dataset = 'birdcamid', size = (640, 640)):
    """
    Resizes an RGB image and uploads image to a RoboFlow dataset
    
    inputs:
        image: RGB cv2 image or numpy array
        dataset: Name of the roboflow dataset 
        size: size to reshape image to
    return:
        r: response of api post
    """

    # resize the image to size
    image = cv2.resize(image, size)
    
    # encode the buffer
    _, buffer = cv2.imencode('.jpg', image)

    # convert buffer to bytes than to ascii
    to_post = base64.b64encode(buffer).decode('ascii')

    # create photo name from date and time
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

    return r

if __name__ == '__main__':
    image = cv2.imread('bus.jpg')
    image = cv2.resize(image, (640, 640))
    from time import time 
    t = time()
    upload_cv2_image(image)
    print(time() - t)
