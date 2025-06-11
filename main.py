# PyTorch-based Facial Recognition Authentication
# Requirements: pip install facenet-pytorch opencv-python torch numpy

import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from datetime import datetime
import winsound
import sys

# Load face detector and embedding model
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load all reference images from 'users' folder
user_folder = 'users'
user_embeddings = []
user_names = []
if not os.path.exists(user_folder):
    os.makedirs(user_folder)
    print(f"Created '{user_folder}' folder. Place reference images inside.")
for fname in os.listdir(user_folder):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(user_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)
        if face is not None:
            emb = resnet(face.unsqueeze(0)).detach().numpy()
            user_embeddings.append(emb)
            user_names.append(os.path.splitext(fname)[0])
if not user_embeddings:
    raise ValueError("No valid user images found in 'users' folder.")
user_embeddings = np.vstack(user_embeddings)

# Logging function
log_file = 'auth_log.txt'
def log_attempt(user, result):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()} - {user} - {result}\n")

# User registration function
def register_user(mtcnn, resnet, user_folder):
    cap = cv2.VideoCapture(0)
    print("Registration mode: Press 'c' to capture, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to capture image from camera.')
            break
        cv2.imshow('Register User', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(rgb)
            if face is not None:
                name = input('Enter username: ')
                save_path = os.path.join(user_folder, f'{name}.jpg')
                cv2.imwrite(save_path, frame)
                print(f'User {name} registered.')
                break
            else:
                print('No face detected. Try again.')
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Similarity threshold
SIMILARITY_THRESHOLD = 0.6

# Authentication loop
cap = cv2.VideoCapture(0)
print('Starting camera for facial authentication...')
authenticated = False
while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to capture image from camera.')
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    box = None
    if face is not None:
        face = face.unsqueeze(0)
        emb = resnet(face).detach().numpy()
        # Cosine similarity for all users
        sims = np.dot(user_embeddings, emb.T) / (np.linalg.norm(user_embeddings, axis=1, keepdims=True) * np.linalg.norm(emb))
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx][0]
        if best_sim > SIMILARITY_THRESHOLD:
            name = user_names[best_idx]
            print(f'Authenticated: {name}!')
            log_attempt(name, 'SUCCESS')
            authenticated = True
            cv2.putText(frame, f'Authenticated: {name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            winsound.Beep(1000, 200)
        else:
            print('Face not recognized.')
            log_attempt('Unknown', 'FAIL')
            cv2.putText(frame, 'Not recognized', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            winsound.Beep(400, 400)
        # Draw rectangle around detected face
        boxes, _ = mtcnn.detect(rgb)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or authenticated:
        break
    elif key == ord('r'):
        cap.release()
        cv2.destroyAllWindows()
        register_user(mtcnn, resnet, user_folder)
        # Reload users after registration
        os.execv(sys.executable, ['python'] + sys.argv)
    elif key == ord('t'):
        try:
            new_thresh = float(input('Enter new similarity threshold (0-1): '))
            SIMILARITY_THRESHOLD = new_thresh
            print(f'New threshold set: {SIMILARITY_THRESHOLD}')
        except Exception:
            print('Invalid threshold.')
cap.release()
cv2.destroyAllWindows()
