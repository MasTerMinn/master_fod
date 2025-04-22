import socket
import cv2
import numpy as np
import json

def create_receiver_socket():
    host = "0.0.0.0"  # รับข้อมูลจากทุก IP
    port = 6000        # พอร์ตเดียวกับที่ส่งข้อมูลมา
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((host, port))
    return server_socket

def receive_data(server_socket):
    try:
        # รับข้อมูล JSON (bounding box)
        message, addr = server_socket.recvfrom(1024)  # รับข้อความ JSON
        bbox_data = json.loads(message.decode())

        # รับข้อมูลภาพ (JPEG)
        frame_data, addr = server_socket.recvfrom(65536)  # รับข้อมูลภาพ (ขนาดสูงสุด 65536 bytes)
        frame = np.frombuffer(frame_data, dtype=np.uint8)  # แปลงข้อมูลเป็น numpy array
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)  # แปลงเป็นภาพ

        # แสดงผลภาพ
        cv2.imshow("Received Frame", frame)
        print("Received frame and bounding box data.")

        return frame, bbox_data

    except Exception as e:
        print(f"Error receiving data: {e}")
        return None, None

# สร้าง socket สำหรับเครื่องรับ
server_socket = create_receiver_socket()

while True:
    # รับข้อมูลจากเครื่องส่ง
    frame, bbox_data = receive_data(server_socket)

    if frame is None:
        break

    # เมื่อได้รับข้อมูลแล้วให้แสดงผล
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการเชื่อมต่อ
server_socket.close()
cv2.destroyAllWindows()

