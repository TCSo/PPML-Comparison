import numpy as np
import onnx 
from concrete.ml.torch.compile import compile_onnx_model

onnx_model = onnx.load("slp.onnx")
onnx.checker.check_model(onnx_model)

input_calib = np.load('input_calib.npy')
input_test = np.load('input.npy')

quantized_module = compile_onnx_model(
    onnx_model, input_calib, n_bits=4
)

input_quantized = quantized_module.quantize_input(input_test)
output_quantized = quantized_module.quantized_forward(input_quantized, fhe='simulate')
output_test = quantized_module.dequantize_output(output_quantized)
# print(output_test)

with open("output_concrete.npy", 'xb') as f:
    np.save(f, output_test)