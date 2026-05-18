import socket
import pickle
import struct
import cv2
import numpy as np

def start_laptop_receiver(ip='0.0.0.0', port=8089):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((ip, port))
    server_socket.listen(10)
    print(f"🚀 Sim2Real Debugger listening on {ip}:{port}...")

    while True:
        conn, addr = server_socket.accept()
        print(f"✅ Connected to Duckiebot: {addr}")
        data = b""
        payload_size = struct.calcsize("Q")

        try:
            while True:
                while len(data) < payload_size:
                    packet = conn.recv(4096)
                    if not packet: break
                    data += packet
                if not data: break

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]
                
                while len(data) < msg_size:
                    data += conn.recv(4096)
                    
                msg_data = data[:msg_size]
                data = data[msg_size:]
                msg = pickle.loads(msg_data)

                img_raw = np.array(msg["image"], dtype=np.uint8)
                
                
                # Convert CHW to HWC for OpenCV
                display_img = np.transpose(img_raw, (1, 2, 0))
                # If it's grayscale (120, 160, 1), squeeze to (120, 160)
                if display_img.shape[2] == 1:
                    display_img = display_img.squeeze()
                display_img = cv2.resize(display_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
                title = "Sim2Real Vision Debug (160x120)"

                
                action = msg.get("action", [0.0, 0.0])
                motors = msg.get("motors", [0.0, 0.0])
                info_text = f"V: {action[0]:.2f} W: {action[1]:.2f} | L/R: {motors[0]:.2f}, {motors[1]:.2f}"
                cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Duckiebot Sim2Real Monitor", display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
                    
        except Exception as e:
            print(f"⚠️ Connection lost: {e}")
        finally:
            conn.close()
            print("🔄 Waiting for Duckiebot reconnection...")

if __name__ == "__main__":
    start_laptop_receiver()