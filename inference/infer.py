import numpy as np
import cv2
from openvino.inference_engine import IECore
from math import sin

YOLOV5N_ANCHORS = [
    [10,13, 16,30, 33,23],  # P3/8
    [30,61, 62,45, 59,119],  # P4/16
    [116,90, 156,198, 373,326],  # P5/32 
    ]

def parse_predictions(preds, threshold = .25, min_edge = .05, img_size = 640):
    """
    parse the 
    """
    # prevent any hazards
    preds = np.copy(preds)

    # filter preds bellow the min confidence threshold
    preds = preds[(preds[..., 4] > threshold)]
    
    # convert xywh to xyxy
    xymin = preds[..., :2] - preds[..., 2:4] / 2 # min = center - width / 2
    xymax = preds[..., :2] + preds[..., 2:4] / 2 # max = center - width / 2

    # clip to 0, 1
    preds[..., 0:2] = np.clip(xymin, 0., 1.) 
    preds[..., 2:4] = np.clip(xymax, 0., 1.) 

    # filter boxes that are too small
    preds = preds[(preds[..., 2] > min_edge)]
    preds = preds[(preds[..., 3] > min_edge)]

    # get max class ids
    class_id = np.expand_dims(np.argmax(preds[..., 5:], axis = -1), axis = -1)

    # concat xmin, ymin, xmax, ymax
    preds = np.concatenate([preds[..., :5], class_id], axis = - 1)

    return preds

def intersection_of_union(box_1, box_2):
    """
    0: xmin, 1: ymin, 2: xmax, 3: ymax
    """
    width_overlap = min(box_1[2], box_2[2]) - max(box_1[0], box_2[0])
    height_overlap = min(box_1[3], box_2[3]) - max(box_1[1], box_2[1])

    # if the height or width of the overlap is 0 or less, the intersection is 0
    if width_overlap <= 0 or height_overlap <= 0: return 0
    area_of_overlap = width_overlap * height_overlap

    # calculate the area of the boxes 
    box_1_area = (box_1[3] - box_1[1]) * (box_1[2] - box_1[0])
    box_2_area = (box_2[3] - box_2[1]) * (box_2[2] - box_2[0])

    # calculate area of union 
    area_of_union = box_1_area + box_2_area - area_of_overlap

    # gaurd against div 0s
    if area_of_union == 0: return 0
    return area_of_overlap / area_of_union

def non_max_surpression(objects, threshold = .5):
    """
    eliminates redundant boxes by checking the interection of union
     - right now this is O(n^2), thats sloopy and should be fixed
     - this could get some easy gains through vectorization
    """
    l = len(objects)

    for i in range(l): 
        # if confidence is too low, skip this iteration
        if objects[i][4] <= 0.0: continue
        # loop over boxes that havent been compared to current box
        for j in range(i + 1, l):
            if intersection_of_union(objects[i], objects[j]) > threshold:
                # set the object with less confidence to 0
                if objects[i][4] > objects[j][4]: 
                    objects[j][4] = 0.
                else: 
                    objects[i][4] = 0.

    # return objects that dont have confidence 0
    return objects[(objects[..., 4] > 0.)] 


def create_detector(num_classes = 80, anchors = YOLOV5N_ANCHORS, img_size = 640):
    num_outputs = num_classes + 5 # num outpus per anchor
    num_anchors = len(anchors[0]) // 2 # num anchors 
    
    def make_grid(nx, ny, i):
        # create a mesh grid from 0 to nx and 0 to ny
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='xy')
        
        # stach xv, yv on the second axis 
        grid = np.stack((xv, yv), axis = 2)
        grid = np.broadcast_to(grid, (1, num_anchors, ny, nx, 2))

        # create the anchor grid, anchors are 'recomender' spots for obj detection nets
        anchor_grid = np.array(anchors[i], dtype=np.float32).reshape((1, num_anchors, 1, 1, 2))
        anchor_grid = np.broadcast_to(anchor_grid,(1, num_anchors, ny, nx, 2))

        return grid, anchor_grid

    def detect(predictions): # call on one image and output all detections
        """
        returns: output of the convnet to confidence ratio
        """
        z = []

        for i, pred in enumerate(predictions): # for each prediction layer
            batch_size, _, ny, nx = pred.shape
            
            stride = img_size // ny

            grid, anchor_grid = make_grid(nx, ny, i)

            # reshape to account for the anchors and out puts 
            x = np.reshape(pred, (batch_size, num_anchors, num_outputs, ny, nx))

            # transpose to (b, num_anchors, ny, nx, num_outputs)
            x = np.transpose(x, (0, 1, 3, 4, 2))

            # sigmoid activation
            y = 1 / (1 + np.exp(-x)) 

            # calculate xy, wh from grid and strides, scale to (0, 1) by dividing by img_size
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + grid) * stride / img_size  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid / img_size # wh

            # reshape y to (b, num_predections, num_outputs)
            y = np.reshape(y, (batch_size, -1, num_outputs))

            # append predictions to z
            z.append(y)
        
        # concatinate z 
        return np.concatenate(z, axis = 1)
        
    return detect

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def prep_cv2_img(img_in, size = 640):
    # prevent hazards 
    img = img_in.copy() 

    # resize image to correct size 
    img = cv2.resize(img, (size, size)) 

    # gbr to rbg
    img = img[..., ::-1] 

    # swap dims to (c, h, w)
    img = np.transpose(img, (2, 0, 1)) 

    # expand dims for batch size of 1, shape (b, c, h, w)
    img = np.expand_dims(img, axis = 0) 

    # properly format as dict from image input 
    return {'images': img} 

def draw_boxes(img, objects, num_classes = 80):
    """
    img: cv2/numpy in rgb order
    objects: numpy of shape (n, 6) where:
        0:4 are xyxy, 4 is confidence, 5 is class
    """
    # get height and width from the first 2 dims of img shape (h, w, c)
    height, width = img.shape[:2]
    # copy the image to prevent hazards 
    result = img.copy()

    factor = 3.14 / num_classes / 2

    for obj in objects:
        # create a color using sin and cos
        color = [int(sin(obj[5] * factor * i * 2 + 1) * 256) for i in range(1, 4)]

        xmin, ymin, xmax, ymax = int(obj[0] * width), int(obj[1] * height), int(obj[2] * width), int(obj[3] * height)

        # plot the rectangle and add a text label bellow
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.putText(result, f"{int(obj[5])} @ {obj[4]:.1%}",
            (xmin, ymin),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5, color, 1, 2)

    return result

class ObjectDetector():
    """
    A nice abstraction of the object detector
    
    inputs a cv2 image (bgr)
    outputs a cv2 image, same size as input (bgr)

    design thoughts:
    i guess this could be a function returning a function rather than a class
    """
    def __init__(self, 
            model_name, 
            device = 'MYRIAD',
            conf_threshold = .2, 
            iou_threshold = .5,
            num_classes = 80,
            anchors = YOLOV5N_ANCHORS,
            img_size = 640
            ):
        # define the intel inference engine
        self.ie = IECore()
        
        # read and load the network onto the device 
        net = self.ie.read_network(model = f'/home/pi/birdnet_torch/models/{model_name}.xml', weights = f'/home/pi/birdnet_torch/models/{model_name}.bin')
        self.net = self.ie.load_network(network = net, device_name = device, num_requests = 2)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # create the network parsing detetor 
        self.detector = create_detector(num_classes = num_classes, anchors = anchors, img_size = img_size)

        self.num_classes = num_classes
    
    def __call__(self, img, return_detected = False):
        """
        return detected: if true will return a boolean of whether or not a bird was detected
        """
        # format the cv2 image for input 
        inputs = prep_cv2_img(img) 

        # infer the networks and convert the dict to a list
        outputs = self.net.infer(inputs=inputs) 
        outputs = [out for _, out in outputs.items()]

        # extract the bounding boxes from the outputs
        preds = self.detector(outputs)

        # parse the predictions, removing any prediction with confidence lower than conf_threshold
        objects = parse_predictions(preds, threshold = self.conf_threshold)

        # apply non_max_surpression, removing any bounding boxes that overlap more than iou_threshold
        objects = non_max_surpression(objects, threshold = self.iou_threshold)

        # draw the bounding boxes with labels 
        img_out = draw_boxes(img, objects, self.num_classes)
        
        if return_detected:
            # return img_out and a boolen representing if objects were detected 
            return img_out, len(objects) > 0
        return img_out

if __name__ == "__main__":
    iou_threshold = .5
    conf_threshold = .4

    model_name = 'yolov5n'
    ie = IECore()
    device = 'CPU'#'MYRIAD'

    net = ie.read_network(model = f'models/{model_name}.xml', weights = f'models/{model_name}.bin')
    exec_net = ie.load_network(network = net, device_name = device, num_requests=2)

    detector = create_detector()  

    cam = cv2.VideoCapture(0)

    while True:
        _, img = cam.read()

        inputs = prep_cv2_img(img)

        outputs = exec_net.infer(inputs=inputs)
        outputs = [out for _, out in outputs.items()]

        preds = detector(outputs)
        objects = parse_predictions(preds, conf_threshold)
        objects = non_max_surpression(objects, threshold = iou_threshold)
        
        img_out = draw_boxes(img, objects)

        cv2.imshow('my webcam', img_out)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        
    cv2.destroyAllWindows()
    