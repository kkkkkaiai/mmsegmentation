import onnx
import onnxruntime as ort
import cv2
import numpy as np

ort_session = ort.InferenceSession('road_seg.onnx')
img = cv2.imread('b1c9c847-3bda4659.jpg')

def prepross(img):
    img = np.array(img, dtype=np.float32).transpose(2,0,1)
    return [img]

img = cv2.imread('b1c9c847-3bda4659.jpg')
img = prepross(img)
outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: img})
print(outputs)

