import socket
import pickle
import struct
import cv2

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
                img = msg["image"]

                # FORCE interpretation as uint8 grayscale
                img = np.array(img, dtype=np.uint8)

                # If it STILL looks inverted, manually flip it to see if it matches Sim
                # img = cv2.bitwise_not(img) 

                cv2.imshow("Duckiebot View (84x84)", img)
                print(f"Action: {msg['action']} | Motors: {msg['motors']}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return
        except Exception as e:
            print(f"Connection lost: {e}")
        finally:
            conn.close()
            print("Waiting for reconnection...")

if __name__ == "__main__":
    start_laptop_receiver()