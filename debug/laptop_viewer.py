import socket
import pickle
import struct
import cv2
import numpy as np

def start_laptop_receiver(ip='0.0.0.0', port=8089):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Reuse port after crash
    server_socket.bind((ip, port))
    server_socket.listen(10)
    print(f"Laptop listening on {ip}:{port}... Waiting for Duckiebot.")

    while True: # Keep the server running even if robot restarts
        conn, addr = server_socket.accept()
        print(f"Connected to Duckiebot at {addr}")
        data = b""
        payload_size = struct.calcsize("Q")

        try:
            while True:
                while len(data) < payload_size:
                    packet = conn.recv(4096)
                    if not packet: break
                    data += packet
                
                if not data: break # Connection closed

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_msg_size)[0]
                
                while len(data) < msg_size:
                    data += conn.recv(4096)
                    
                msg_data = data[:msg_size]
                data = data[msg_size:]
                
                msg = pickle.loads(msg_data)
                img_stack = msg["image"]
                action = msg["action"]
                motors = msg["motors"]
                
                img_stack = np.array(img_stack, np.uint8)
                combined_view = np.hstack([img_stack[i] for i in range(img_stack.shape[0])])                

                display_img = cv2.resize(combined_view, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

                for i in range(1, 4):
                    cv2.line(display_img, (84*4*i, 0), (84*4*i, 84*4), (255), 1)

                action_text = f"V: {action[0]:.2f} Omega: {action[1]:.2f} | Motors L/R: [{motors[0]:.2f}, {motors[1]:.2f}]"
                cv2.putText(display_img, action_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
                
                cv2.imshow("Duckiebot: Oldest Frame (Left) -> Newest Frame (Right)", display_img)
                print(f"\r{action_text}", end="", flush=True)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
        except Exception as e:
            print(f"Connection lost: {e}")
        finally:
            conn.close()
            print("Waiting for reconnection...")

if __name__ == "__main__":
    start_laptop_receiver()