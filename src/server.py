import socket
import cv2
import socket
import pickle
import numpy as np
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MAX_LENGTH = 65536
start_time = time.time()
interval   = 1
counter    = 0

# Load pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("./face_database.xml")  # Replace 'path_to_your_trained_recognizer.xml' with the actual path

# Function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Example usage
def main():
    # Load image captured by ESP32-CAM (replace 'image.jpg' with the actual image filename)
    image = cv2.imread(r'./student_images/Jane_Doe.jpg')

    # Detect faces in the image
    faces = detect_faces(image)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def server_program(): 
    # get the hostname
    host = socket.gethostname()
    port = 5000  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(2)
    conn, address = server_socket.accept()  # accept new connection
    print("Connection from: " + str(address))
    while True:
        buffer   = conn.recv(MAX_LENGTH)
        frame = np.frombuffer(buffer, dtype=np.uint8)
        frame = frame.reshape(1, frame.shape[0])

        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        if frame is not None:
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                label, _ = recognizer.predict(gray_face)
                cv2.putText(frame, f'Label: {label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            time.sleep(0.09)
            global counter
            global start_time
            counter += 1
            if (time.time() - start_time) > interval:
                print("FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()
            if frame is not None and type(frame) == np.ndarray:
                cv2.imshow("Stream", frame)
                if cv2.waitKey(1) == 27:
                    break

    conn.close()  # close the connection
    
# server_program()

@app.route('/start_server', methods=['POST'])
def start_server():
    server_thread = threading.Thread(target=server_program)
    server_thread.start()
    return jsonify({"status": "Server started"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
