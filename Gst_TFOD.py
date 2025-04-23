import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load your TFLite model
interpreter = tflite.Interpreter(model_path="/home/mendel/tflite_test/detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# GStreamer pipeline to send video
gst_str = (
    "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast ! "
    "rtph264pay config-interval=1 pt=96 ! udpsink host=10.235.175.5 port=5000"
)
out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30, (640, 480), True)

# Camera input
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

class_names = ['Bolt', 'Wire', 'SodaCan', 'Hammer', 'Wrench']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_img = cv2.resize(frame, (300, 300))
    input_tensor = np.expand_dims(input_img.astype(np.float32) / 255.0, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    bbox = interpreter.get_tensor(output_details[0]['index'])[0]  # [x_min, y_min, x_max, y_max]
    class_probs = interpreter.get_tensor(output_details[1]['index'])[0]
    class_idx = np.argmax(class_probs)
    confidence = float(class_probs[class_idx])

    if confidence > 0.6:
        # Scale bbox back to original size
        x_min = int(bbox[0] * (640 / 300))
        y_min = int(bbox[1] * (480 / 300))
        x_max = int(bbox[2] * (640 / 300))
        y_max = int(bbox[3] * (480 / 300))

        label = f"{class_names[class_idx]} ({confidence:.2f})"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Write to GStreamer pipeline
    out.write(frame)

# Cleanup
cap.release()
out.release()
