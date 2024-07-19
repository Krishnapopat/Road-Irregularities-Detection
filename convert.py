import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import time

def load_engine(engine_file_path):
    print(f"Loading TensorRT engine: {engine_file_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess_image(image, input_shape):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_shape[3], input_shape[2]))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return np.ascontiguousarray(image)

def process_results(output, orig_h, orig_w, input_shape, conf_threshold=0.25, iou_threshold=0.45):
    output = output.reshape(-1, output.shape[-1])
    mask = output[:, 4] > conf_threshold
    detections = output[mask]
    
    if len(detections) == 0:
        return [], [], []
    
    input_h, input_w = input_shape[2:]
    scale_h, scale_w = orig_h / input_h, orig_w / input_w
    
    boxes = detections[:, :4]
    scores = detections[:, 4]
    class_ids = detections[:, 5]
    
    boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * scale_w
    boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * scale_h
    boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) * scale_w
    boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) * scale_h
    
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    
    return boxes[indices], scores[indices], class_ids[indices]

def draw_detections(image, boxes, scores, class_ids, class_names):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{class_names[int(class_id)]}: {score:.2f}"
        
        color = (0, 255, 0)  # Green color for the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image

# Main execution
if __name__ == "__main__":
    print(f"TensorRT version: {trt.__version__}")

    engine_path = "best.engine"
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    input_tensor_name = engine.get_tensor_name(0)
    output_tensor_name = engine.get_tensor_name(1)
    input_shape = engine.get_tensor_shape(input_tensor_name)
    output_shape = engine.get_tensor_shape(output_tensor_name)

    input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
    output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    stream = cuda.Stream()

    # List of class names for your model
    class_names = ["openmanhole", "pothole", "unmarkedbump"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to read camera feed")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        img = preprocess_image(frame, input_shape)

        # Run inference
        cuda.memcpy_htod_async(d_input, img, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        # Process results
        boxes, scores, class_ids = process_results(output, orig_h, orig_w, input_shape)

        # Draw detections
        annotated_image = draw_detections(frame, boxes, scores, class_ids, class_names)

        cv2.imshow('Webcam', annotated_image)
        
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()