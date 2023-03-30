from pathlib import Path

import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type

import numpy as np
import cv2


SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4


def resize_and_pad(image, new_shape):
    old_size = image.shape[:2] 
    ratio = float(new_shape[-1]/max(old_size))#fix to accept also rectangular images
    new_size = tuple([int(x*ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))
    
    delta_w = new_shape[1] - new_size[1]
    delta_h = new_shape[0] - new_size[0]
    
    color = [100, 100, 100]
    new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)
    
    return new_im, delta_w, delta_h


def main( ):

    # Step 1. Initialize OpenVINO Runtime core
    core = ov.Core()
    # Step 2. Read a model
    model = core.read_model(str(Path("../model/yolov5n.xml")))

   
    # Step 3. Read input image
    img = cv2.imread(str(Path("../imgs/000000000312.jpg")))
    # resize image
    img_resized, dw, dh = resize_and_pad(img, (640, 640))


    # Step 4. Inizialize Preprocessing for the model
    ppp = PrePostProcessor(model)
    # Specify input image format
    ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
    #  Specify preprocess pipeline to input image without resizing
    ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
    # Specify model's input layout
    ppp.input().model().set_layout(Layout("NCHW"))
    #  Specify output results format
    ppp.output().tensor().set_element_type(Type.f32)
    # Embed above steps in the graph
    model = ppp.build()
    compiled_model = core.compile_model(model, "CPU")


    # Step 5. Create tensor from image
    input_tensor = np.expand_dims(img_resized, 0)


    # Step 6. Create an infer request for model inference 
    infer_request = compiled_model.create_infer_request()
    infer_request.infer({0: input_tensor})


    # Step 7. Retrieve inference results 
    output = infer_request.get_output_tensor()
    detections = output.data[0]


    # Step 8. Postprocessing including NMS  
    boxes = []
    class_ids = []
    confidences = []
    for prediction in detections:
        confidence = prediction[4].item()
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = prediction[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()
                xmin = x - (w / 2)
                ymin = y - (h / 2)
                box = np.array([xmin, ymin, w, h])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

    detections = []
    for i in indexes:
        j = i.item()
        detections.append({"class_index": class_ids[j], "confidence": confidences[j], "box": boxes[j]})


    # Step 9. Print results and save Figure with detections
    for detection in detections:
    
        box = detection["box"]
        classId = detection["class_index"]
        confidence = detection["confidence"]

        rx = img.shape[1] / (img_resized.shape[1] - dw)
        ry = img.shape[0] / (img_resized.shape[0] - dh)
        box[0] = rx * box[0]
        box[1] = ry * box[1]
        box[2] = rx * box[2]
        box[3] = ry * box[3]

        print( f"Bbox {i} Class: {classId} Confidence: {confidence} Scaled coords: [ cx: {(box[0] + (box[2] / 2)) / img.shape[1]}, cy: {(box[1] + (box[3] / 2)) / img.shape[0]}, w: {box[2]/ img.shape[1]}, h: {box[3] / img.shape[0]} ]" )
        xmax = box[0] + box[2]
        ymax = box[1] + box[3]
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(xmax), int(ymax)), (0, 255, 0), 3)
        img = cv2.rectangle(img, (int(box[0]), int(box[1]) - 20), (int(xmax), int(box[1])), (0, 255, 0), cv2.FILLED)
        img = cv2.putText(img, str(classId), (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imwrite("./detection_python.png", img)
    
if __name__ == '__main__':

    main( )