# Birdnet-Edge 
## Inference and Data Collection on the edge, Learning in the cloud.

Birdnet-Edge is a circularly learning system consisting of three main parts; inference, data labeling, and training. Inference is done on the edge, datalabing is done via Roboflow's web service, and learning is done in the cloud. 

A machine learning model was initally train on DATASET, a collection of NUMBER images of NUMBER OF bird species. The network achieved METRIC idenidfying any bird, so importantly the network can idetify and locate a bird out of an image with very high accuracy. The problem isnt locating a bird, It is classifing the bird by species.  

To solve this, more data must be collected. Hence, an object detection model is deployed on the edge in front of a bird feeder. If a bird is detected in the image, the image is then sent to Roboflows labeling service through there convient API. There, the photo is labeled by an expert, my father. A Cloud server than fetches the images from roboflow, agian using the well designed Roboflow API. A model is trained in the cloud, learning on the nearly collected and labeled photos. Then the updated model is retrieved from the cloud to do inference, starting the entire process over.

Although this project is specific to birds the potential of the system goes far beyond avian follery. The system could easily scale out across multiple locations, collecting from a deverese range of habitiats. Furthermore, the system can be ablied to an arbitrary amount of problems. For example, it could be combined with reinforcement learning to optimize traffic flow and pedesterain saftery through an intersection.

## Tech used
- Infernence 
    - Object Detection Algorithm: YoloV5n 
    - Inference framework: Intel OpenVino
    - General Purpose Compute: RaspberryPi 3B+ 
    - Camera: Arducam 
    - Acceloration: Intel Neural Compute Stick 2
- Data Labeling
    - Roboflow 
- Training
    - Server: AWS SERVERTYPE with Intel Habanna deep learning acceloration 
    - ML Framework: PyTorch 

## Why Edge matters
It is a fair question to ask, why not just send the photos to the cloud? In some situations, doing inference in the cloud is best. Perhaps you have low inference volumn or you're service is already in the cloud. Cloud computing is pretty cheep, easily scaled, and you don't need to own any depriecated hardware. However, cloud computing and machine learning is growing in compute at a rapid rate RATE HERE. Simimalry, Plus, increased wireless transmission rate are going to further increase the demand for cloud servers and clog backhaul infrastructure. While radio communication is increasing with 5G, the core network of 5G, where cell towers and servers communcate through fiber optics and wire,  is built on the core network of the 4G LTE system. All of this will increase the costs of cloud computing and add latency to communication between you and the cloud.

Simply, Edge matters becuase it is needed to take advatage of machine when latency and data volume is an issue. For example, any mission critical systems will not be able to deal with latency at the edge. Likewise, any application that requires inference on significantly large data will be bottle necked by backhaul speeds. 

## Where does Cloud fit in
Training is a very compute heavy task that can significantly benifit from parrellism gained through batch processing multiple images together. Additonaly, after each epoch the weights in the neural network must be updated so the network can improve apon what its previous solution. 
Federated learning, where training is done on the edge across many devices, is showing some promise in certain applications. However, sending weights between distributed edge devices will succum to the same backhaul bottleneck that limits inference on the cloud. Hence, training in a centralized location where data is accessable and latency is low provides a siginifacnt advatage. A key point is that not all data is labeled and sent to the cloud, only data the inference step deams important enough to be labeled. 

## Cloud and Edge TLDR
Communication across the 'backhaul' can limit the amount of data you can send between devices in terms of cost and time. Inference and data collection need to happen over distrubted area in sometimes time senistive situation. Therefore, doing inference on the edge and communicating only important information to the server makes sense. Training consists of quickly iterating through data multiple and updating large tensors. Therefor, it makes sense to collect the data in a central cloud server and train while taking advatage of local storage and parrellism. 

## Dataflow
INSERT BLOCK DIAGRAM

## Metrics
 show improving accuracy, recall, precision

## OpenVino 
OpenVinos is intels inference API/Framework that allows you to do inference on the neural compute stick. The API abstracts the hardware away so you can use the same code tos do inference on any Intel processor that has OpenVino support. This is especially nice becuase you can debug logic on your Intel CPU before you deploy the network onto the edge device. 

Setting up OpenVino on the RaspberryPi is pretty straight forward as long as you have the Buster distro of Raspbian, this caused me quite a headache and a few hours. Just follow the directions from the [OpenVino Docs](ADD LINK). 

## Yolov5n
Yolov5n is a very popular one shot object detection algorithm. It is the smallest of the networks in its gernation with less than 2 million parameters, making inference extra fast. Sadley, The Myraid processor on the Neural Compute Stick does not spport some of the operations near the output of the model. Luckily for you, these operations are not too compute heavy and I took the time to write code to turn the output of the last convolutional layer into workable predictions. 

With USB 3 on my laptop the Neural compute stick can do inference on an image in ~ 140 ms.  This is lowered on the RaspberryPi 3 since it uses USB 2, creating a IO bottle neck. A RaspberryPi 4 should be able to hit a similar inference as mhy laptop since it uses USB 3. I am not too concerned with realtime inference since birds stay at a bird feeder for more than a few seconds at a time. 

Compiling the model from torch to openvino is not too bad, you can use my tools in training/openvino_utils.py. There are a few intermediate steps that I have already taken care of. 

## Roboflow 
Roboflow is a SaaS (Software as a service) start up providing some great tools for AI. If you wish to outsource inference or training they have a service for that. For some companies, that makes a lot of sense. Additionally, they have tools that can make the end to end machine learning process a lot easier. I used their labeling service to streamline the process. Simply, I send photos that contian birds to a 
## Training 

## Intel Habana on AWS
Talk about stats and what the habana does 

## 
