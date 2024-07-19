# from ultralytics import YOLOv10
# from ultralytics.engine.results import Results
# import cv2
# from typing import List

# # Load the YOLOv10 model
# model = YOLOv10('best.pt')  # Assuming 'best.pt' is your trained model weights file

# # Function to predict using the model
# def predict(image_path):
#     results : List[Results] = model(image_path)
#     results[0].show()
#     results[0].save()
#     print(results)
    

# # Example usage
# image_path = '4.jpg'
# predict(image_path)

# import cv2
# import supervision as sv
# from ultralytics import YOLOv10
# import os

# model = YOLOv10('best.pt')

# bounding_box_annotator = sv.BoundingBoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# cap=cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Unable to read Camera feed")


# img_counter = 0 
# while True:
#     ret, frame=cap.read()
#     if not ret:
#         break

#     results = model(frame)[0]
#     detections = sv.Detections.from_ultralytics(results)

#     annotated_image = bounding_box_annotator.annotate(
#     scene=frame, detections=detections)
#     annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections)
#     cv2.imshow('Webcam',annotated_image)
#     k=cv2.waitKey(1)

#     if k%256==27:
#         print("Escape hit, closing ... ")
#         break

# cap.release()
# cv2.destroyAllWindows()


# model = YOLOv10('best.pt')
# image = cv2.imread('4.jpg')
# results = model(image)[0]
# detections = sv.Detections.from_ultralytics(results)

# bounding_box_annotator = sv.BoundingBoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# annotated_image = bounding_box_annotator.annotate(
#     scene=image, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections)

# sv.plot_image(annotated_image)

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    with builder.build_engine(network, config) as engine:
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

onnx_file_path = "best.onnx"
engine_file_path = "best.engine"

engine = build_engine(onnx_file_path, engine_file_path)
if engine:
    print("TensorRT engine built successfully")
else:
    print("Failed to build TensorRT engine")
