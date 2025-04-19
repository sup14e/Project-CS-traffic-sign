import cv2
import torch
import torchvision.transforms as T
from torchvision.ops import box_convert
from model import Model4
import numpy as np

# Define your label map
label_map = {
    0: "stop", 1: "yield", 2: "yieldAhead", 3: "merge",
    4: "signalAhead", 5: "pedestrianCrossing", 6: "keepRight",
    7: "speedLimit35", 8: "speedLimit25"
}

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model4().to(device)
model.load_state_dict(torch.load("model_best_vloss.pth"))
model.eval()

# Transform for input image
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Start video capture
cap = cv2.VideoCapture(0)

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Store original frame size
        original_h, original_w = frame.shape[:2]

        # Preprocess the frame
        input_img = transform(frame).unsqueeze(0).to(device)

        # Predict
        class_logits, bbox_pred = model(input_img)
        class_idx = torch.argmax(class_logits, dim=1).item()
        class_label = label_map[class_idx]

        # Convert bbox from cxcywh to xyxy
        bbox_xyxy = box_convert(bbox_pred, in_fmt="cxcywh", out_fmt="xyxy").squeeze().cpu().numpy()

        # Rescale bbox to original frame size
        bbox_xyxy[0::2] *= original_w
        bbox_xyxy[1::2] *= original_h
        x1, y1, x2, y2 = bbox_xyxy.astype(int)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Traffic Sign Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
