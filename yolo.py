import cv2
import numpy as np

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = '{} {:,.2%}'.format(classes[class_id], confidence)
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,0,255), 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

classes = None
with open("apple/classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet("apple/apple.weights", "apple/apple.cfg")

cap = cv2.VideoCapture(0)

while cv2.waitKey(1) < 0 or False:
    
    hasframe, image = cap.read()
    image=cv2.resize(image, (800, 600)) 
    
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
    Width = image.shape[1]
    Height = image.shape[0]
    net.setInput(blob)
    
    outs = net.forward(getOutputsNames(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    for out in outs: 
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    
    cv2.imshow("Yolo" , image)
