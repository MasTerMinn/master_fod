import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import socket
import json

def create_socket():
    host = "10.235.175.5" 
    port = 6000
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return client_socket, host, port

def send_data(client_socket, host, port, frame, bbox_data):
    try:
        # Compress frame as JPEG
        _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_bytes = encoded_frame.tobytes()

        # Convert bbox data to JSON
        message = json.dumps(bbox_data)

        # Send JSON first
        client_socket.sendto(message.encode(), (host, port))

        # Send frame
        client_socket.sendto(frame_bytes, (host, port))
        
        print("Sent frame and bounding box data.")

    except Exception as e:
        print(f"Error sending data: {e}")

# Load the TFLite model
MODEL_PATH = "/home/mendel/tflite_test/detect.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the target input shape (300x300)
height, width = 300, 300

# Define class names
class_names = ['Bolt', 'Wire', 'SodaCan', 'Hammer', 'Wrench']

# Open Webcam
cap = cv2.VideoCapture(2)

# Set the highest available resolution (640x480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

client_socket, host, port = create_socket()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to match TFLite input shape (300x300)
    img = cv2.resize(frame, (width, height))
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Get predictions
    output_1 = interpreter.get_tensor(output_details[0]['index'])  # Bounding box [1, 4]
    output_2 = interpreter.get_tensor(output_details[1]['index'])  # Class probabilities [1, 5]

    bbox = output_1[0]  # [x_min, y_min, x_max, y_max]
    class_probs = output_2[0]
    class_idx = np.argmax(class_probs)
    confidence = float(class_probs[class_idx])  # Convert to standard float

    bbox_data = {}

    if confidence > 0.6:
        # Scale back to 640x480
        x_min = int(bbox[0] * (640 / 300))
        y_min = int(bbox[1] * (480 / 300))
        x_max = int(bbox[2] * (640 / 300))
        y_max = int(bbox[3] * (480 / 300))

        # Draw bounding box on the frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw class label
        class_label = f"{class_names[class_idx]} ({confidence:.2f})"
        cv2.putText(frame, class_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Send data to the other machine
    send_data(client_socket, host, port, frame, bbox_data)

# Clean up
cap.release()
client_socket.close()

