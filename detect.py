import cv2
import numpy as np
import os
import imutils
import torch
from torchvision import transforms
from torch.nn.functional import softmax

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Correct paths to YOLO files
yolo_cfg_path = 'yolov3.cfg'
yolo_weights_path = 'yolov3.weights'

# Load YOLO network
net = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load helmet detection PyTorch model
model = torch.load('best.pt', map_location=torch.device('cpu'))  # Load the PyTorch model
model.eval()  # Set the model to evaluation mode
print('Model loaded!!!')

cap = cv2.VideoCapture('input_video.mp4')
COLORS = [(0, 255, 0), (0, 0, 255)]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter('output.avi', fourcc, 5, (888, 500))

# Transform to apply to the input image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.cvtColor(helmet_roi, cv2.COLOR_BGR2RGB)
        helmet_roi = transform(helmet_roi)
        helmet_roi = helmet_roi.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(helmet_roi)
            probabilities = softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities.data, 1)
        return int(predicted.item())
    except Exception as e:
        print(f"Error in helmet_or_nohelmet: {e}")
        return None

ret = True

while ret:
    ret, img = cap.read()
    if not ret:
        break
    
    img = imutils.resize(img, height=500)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    classIds = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = [int(c) for c in COLORS[classIds[i]]]
            if classIds[i] == 0:  # bike
                helmet_roi = img[max(0, y):max(0, y) + max(0, h) // 4, max(0, x):max(0, x) + max(0, w)]
            else:  # number plate
                x_h = x - 60
                y_h = y - 350
                w_h = w + 100
                h_h = h + 100
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                if y_h > 0 and x_h > 0:
                    h_r = img[y_h:y_h + h_h, x_h:x_h + w_h]
                    c = helmet_or_nohelmet(h_r)
                    if c is not None:
                        cv2.putText(img, ['helmet', 'no-helmet'][c], (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h), (255, 0, 0), 10)

    writer.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break

writer.release()
cap.release()
cv2.destroyAllWindows()
