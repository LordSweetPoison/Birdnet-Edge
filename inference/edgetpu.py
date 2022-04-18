import os
import numpy as np
import pycoral.utils.edgetpu as etpu
from pycoral.adapters import common

class EdgeTPUModel:
    def __init__(self, model_file):
        """Creates an object for running a Yolov5 model on an EdgeTPU
       
        model_file: path to edgetpu-compiled tflite file
        """
    
        model_file = os.path.abspath(model_file)
    
        if not model_file.endswith('tflite'):
            model_file += ".tflite"
            
        self.model_file = model_file
        
        self.make_interpreter()
        self.input_size = common.input_size(self.interpreter)
        
    
    def make_interpreter(self):
        """
        Internal function that loads the tflite file and creates
        the interpreter that deals with the EdgetPU hardware.
        """
        # Load the model and allocate
        self.interpreter = etpu.make_interpreter(self.model_file)
        self.interpreter.allocate_tensors()
    
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_zero = self.input_details[0]['quantization'][1]
        self.input_scale = self.input_details[0]['quantization'][0]
        self.output_zero = self.output_details[0]['quantization'][1]
        self.output_scale = self.output_details[0]['quantization'][0]
        
        # If the model isn't quantized then these should be zero
        # Check against small epsilon to avoid comparing float/int
        if self.input_scale < 1e-9:
            self.input_scale = 1.0
        
        if self.output_scale < 1e-9:
            self.output_scale = 1.0
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Predict function using the EdgeTPU
        Inputs:
            image: (C, H, W) image tensor
            with_nms: apply NMS on output
        Returns:
            prediction array (with or without NMS applied)
        """
        
        # Transpose if C, H, W
        if image.shape[0] == 3:
          image = image.transpose((1,2,0))
        
        x = image.astype('float32')

        # Scale input, conversion is: real = (int_8 - zero)*scale
        x = (x/self.input_scale) + self.input_zero
        x = x[np.newaxis].astype(np.uint8)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        
        # Scale output
        result = (common.output_tensor(self.interpreter, 0).astype('float32') - self.output_zero) * self.output_scale
  
        return result
          
    
    
    