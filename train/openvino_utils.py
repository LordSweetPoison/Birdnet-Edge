from setuptools import setup
import torchvision.models as models
import torchvision
import torch 
import onnx
import onnxruntime as ort
import numpy as np
from openvino.inference_engine import IECore
import timm
import cv2
import onnxsim
import subprocess
import os 

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def torch_to_onnx(
    model, 
    onnx_model_path,
    batch_size,
    input_shape, 
    simplify = True,
    dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1, 3, 640, 640)
                    'output': {0: 'batch', 1: 'anchors'}}  # shape(1, 25200, 85)
    ):

    model.eval() # set model to eval mode

    dummy_input = torch.randn(batch_size, *input_shape, requires_grad = True)

    torch.onnx.export(model, # model being run
                  dummy_input, # model input (or a tuple for multiple inputs)
                  onnx_model_path, # where to save the model (can be a file or file-like object)
                  training = False,
                  export_params = True, # store the trained parameter weights inside the model file
                  opset_version = 11, # the ONNX version to export the model to
                  do_constant_folding = True, # whether to execute constant folding for optimization
                  input_names = ['images'], # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes = dynamic_axes)
    
    print(f"Onnx model saved to {onnx_model_path}")

    if simplify:
        model_onnx = onnx.load(onnx_model_path)
        model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape = True,
                    input_shapes = {'images': [batch_size, *input_shape]})
                    
        assert check, 'simplified check failed'
        onnx.save(model_onnx, onnx_model_path)
        print('simplifed model saved')


def load_onnx_model(path):
    # load the model from the path
    model = onnx.load(path)
    # check the model
    onnx.checker.check_model(model)

    return model

def test_onnx_model(torch_model, onnx_model_path, batch_size, input_shape):
    ort_session = ort.InferenceSession(onnx_model_path)

    # define model input x to test onx vs torch
    inputs = torch.randn(batch_size, *input_shape)

    torch_out = torch_model(inputs)[0]

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0][0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

def compile_openvino_model(openvino_mopy_path,
        openvino_path, 
        onnx_model_path,
        output_dir, 
        input_shape = (1, 3, 640, 640),
        output_layers = ['Conv_199', 'Conv_510', 'Conv_821'], # defualt in yolov5n is used
        data_type = 'FP16',
        device = 'MYRIAD', # device to 
        scale = 255 # amount to scale inputs, this converts to 255
        ):

    # set up env variables 
    setupvars_cmd = os.path.join(openvino_path, 'bin', 'setupvars.bat')

    print('setting up openvino vars')

    process = subprocess.Popen(setupvars_cmd, stdout=subprocess.PIPE)
    output, error = process.communicate()

    print(output)

    ie = IECore()
    model = ie.read_network(model = onnx_model_path)
    versions = ie.get_versions(device)

    print(versions)

    print(f"{device}")
    print(f"MKLDNNPlugin version ......... {versions[device].major}.{versions[device].minor}")
    print(f"Build ........... {versions[device].build_number}")

    # create bash command to compile model using openvino
    bash_command = f'''python?{openvino_mopy_path}?--input_model?{onnx_model_path}?--output_dir?{output_dir}?--input_shape?{str(input_shape).replace(" ", "")}?--output?{",".join(output_layers)}?--data_type?{data_type}?--scale?{scale}'''.split(sep = '?') # '?' is used as a seperator since paths may have spaces 

    # compile model
    print(bash_command)
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    return output, error

if __name__ == "__main__":
    INPUT_SHAPE = (3, 640, 640)
    batch_size = 1

    model_name = 'yolov5n'
    onnx_model_path = f'{model_name}.onnx'
    onnx_model_path = "C:\\Users\\14135\\Desktop\\birdnet_torch\\yolov5n.onnx"

    """
    print('loading torch model')

    torch_model =  torch.hub.load('ultralytics/yolov5', 'yolov5n') #models.mobilenet_v3_large()
    torch_model.eval()

    print('converting roch to onnx')
    torch_to_onnx(torch_model, onnx_model_path, batch_size, INPUT_SHAPE)
    """

    print('compiling openvino model')
    mopy_path = 'C:\\Program Files (x86)\\Intel\\openvino_2021\\deployment_tools\\model_optimizer\\mo.py'
    compile_openvino_model(mopy_path, 'C:\\Program Files (x86)\\Intel\\openvino_2021', onnx_model_path, "C:\\Users\\14135\\Desktop\\Birdnet-Edge\\models")
