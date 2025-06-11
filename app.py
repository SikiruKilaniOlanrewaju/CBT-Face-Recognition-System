from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
USER_FOLDER = 'users'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SIMILARITY_THRESHOLD = 0.6

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(USER_FOLDER):
    os.makedirs(USER_FOLDER)

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_user_embeddings():
    user_embeddings = []
    user_names = []
    for fname in os.listdir(USER_FOLDER):
        if fname.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            img_path = os.path.join(USER_FOLDER, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(rgb)
            if face is not None:
                emb = resnet(face.unsqueeze(0)).detach().numpy()
                user_embeddings.append(emb)
                user_names.append(os.path.splitext(fname)[0])
    if user_embeddings:
        user_embeddings = np.vstack(user_embeddings)
    return user_embeddings, user_names

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            # Authenticate
            img = cv2.imread(filepath)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(rgb)
            if face is None:
                flash('No face detected in uploaded image.')
                return redirect(request.url)
            face = face.unsqueeze(0)
            emb = resnet(face).detach().numpy()
            user_embeddings, user_names = load_user_embeddings()
            if user_embeddings is None or len(user_embeddings) == 0:
                flash('No registered users found.')
                return redirect(request.url)
            sims = np.dot(user_embeddings, emb.T) / (np.linalg.norm(user_embeddings, axis=1, keepdims=True) * np.linalg.norm(emb))
            best_idx = np.argmax(sims)
            best_sim = sims[best_idx][0]
            if best_sim > SIMILARITY_THRESHOLD:
                name = user_names[best_idx]
                flash(f'Authenticated: {name}! (Similarity: {best_sim:.2f})')
            else:
                flash('Face not recognized.')
            return redirect(request.url)
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if 'file' not in request.files or 'username' not in request.form:
            flash('Missing file or username')
            return redirect(request.url)
        file = request.files['file']
        username = request.form['username']
        if file.filename == '' or username.strip() == '':
            flash('No selected file or username')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(username + '.' + file.filename.rsplit('.', 1)[1].lower())
            filepath = os.path.join(USER_FOLDER, filename)
            file.save(filepath)
            flash(f'User {username} registered!')
            return redirect(url_for('register'))
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
