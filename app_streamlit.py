import streamlit as st
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from datetime import datetime
from PIL import Image
import random
import time
import json

USER_FOLDER = 'users'
LOG_FILE = 'auth_log.txt'
SIMILARITY_THRESHOLD = 0.6
USER_TIME_FILE = 'user_time_limits.json'

# --- Helper for per-user time limit ---
def get_user_time_limit(username, default=120):
    if os.path.exists(USER_TIME_FILE):
        with open(USER_TIME_FILE, 'r') as f:
            data = json.load(f)
        return data.get(username, default)
    return default

def set_user_time_limit(username, seconds):
    data = {}
    if os.path.exists(USER_TIME_FILE):
        with open(USER_TIME_FILE, 'r') as f:
            data = json.load(f)
    data[username] = seconds
    with open(USER_TIME_FILE, 'w') as f:
        json.dump(data, f)

# --- Manual reset for attempts ---
def reset_user_attempt(username):
    if os.path.exists('cbt_results.txt'):
        with open('cbt_results.txt', 'r') as f:
            lines = f.readlines()
        with open('cbt_results.txt', 'w') as f:
            for line in lines:
                if username not in line:
                    f.write(line)

# --- Utility Functions ---
def log_attempt(user, result):
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.now()} - {user} - {result}\n")

def load_users(user_folder, mtcnn, resnet):
    user_embeddings = []
    user_names = []
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
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
        return None, None
    user_embeddings = np.vstack(user_embeddings)
    return user_embeddings, user_names

def save_user_image(username, image):
    save_path = os.path.join(USER_FOLDER, f'{username}.jpg')
    image.save(save_path)
    return save_path

def get_face_embedding(image, mtcnn, resnet):
    img = np.array(image.convert('RGB'))
    face = mtcnn(img)
    if face is not None:
        emb = resnet(face.unsqueeze(0)).detach().numpy()
        return emb
    return None

# --- CBT Questions (example) ---
CBT_QUESTIONS = [
    {
        'question': 'Which data structure uses FIFO (First In, First Out) principle?',
        'options': ['Stack', 'Queue', 'Tree', 'Graph'],
        'answer': 1
    },
    {
        'question': 'What does CPU stand for?',
        'options': ['Central Processing Unit', 'Computer Personal Unit', 'Central Programming Unit', 'Control Processing Unit'],
        'answer': 0
    },
    {
        'question': 'Which of the following is NOT an operating system?',
        'options': ['Linux', 'Windows', 'Oracle', 'macOS'],
        'answer': 2
    },
    {
        'question': 'What is the time complexity of binary search in a sorted array?',
        'options': ['O(n)', 'O(log n)', 'O(n^2)', 'O(1)'],
        'answer': 1
    },
    {
        'question': 'Which language is primarily used for web development?',
        'options': ['Python', 'C++', 'HTML', 'Java'],
        'answer': 2
    },
    {
        'question': 'Who is known as the father of computers?',
        'options': ['Charles Babbage', 'Alan Turing', 'Bill Gates', 'Ada Lovelace'],
        'answer': 0
    },
    {
        'question': 'Which of the following is a relational database management system?',
        'options': ['MySQL', 'MongoDB', 'Redis', 'Neo4j'],
        'answer': 0
    },
    {
        'question': 'What does RAM stand for?',
        'options': ['Read Access Memory', 'Random Access Memory', 'Run Access Memory', 'Read And Memory'],
        'answer': 1
    },
    {
        'question': 'Which protocol is used to transfer web pages?',
        'options': ['FTP', 'SMTP', 'HTTP', 'SSH'],
        'answer': 2
    },
    {
        'question': 'Which of the following is used for version control?',
        'options': ['Git', 'Java', 'Linux', 'HTML'],
        'answer': 0
    }
]

# --- Model Initialization ---
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

st.set_page_config(page_title="University of Ilorin CBT Portal with Facial Recognition", page_icon="🧑‍💻", layout="centered")

menu = ["Welcome", "Authenticate & Start CBT", "Register", "Take Exam", "Admin Login"]
choice = st.sidebar.selectbox("Menu", menu)

# --- Admin Authentication State ---
if 'admin_authenticated' not in st.session_state:
    st.session_state['admin_authenticated'] = False
ADMIN_PASSWORD = "cbtadmin2025"  # Change this to a strong password in production

# --- Admin Login View ---
if choice == "Admin Login" and not st.session_state['admin_authenticated']:
    st.title("Admin Login")
    admin_pass = st.text_input("Enter admin password", type="password")
    if st.button("Login"):
        if admin_pass == ADMIN_PASSWORD:
            st.session_state['admin_authenticated'] = True
            st.success("Admin login successful!")
            st.rerun()
        else:
            st.error("Incorrect password.")

# --- Admin Dashboard (Secured) ---
if choice == "Admin Login" and st.session_state['admin_authenticated']:
    st.title("Admin Dashboard")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Analytics", "PDF Export", "User Management", "Time Limits", "Manual Reset", "Exam Questions", "Results"
    ])

    # --- Analytics Tab ---
    with tab1:
        st.subheader("Authentication Analytics")
        import matplotlib.pyplot as plt
        import pandas as pd
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                logs = [line.strip() for line in f if line.strip()]
            if logs:
                # Example log format: '2024-06-01 12:00:00 - user1 - SUCCESS'
                data = [l.split('-') for l in logs]
                # Clean up and only use the last 3 fields (strip whitespace)
                data = [[x.strip() for x in row[-3:]] for row in data if len(row) >= 3]
                df = pd.DataFrame(data, columns=["Timestamp", "Username", "Status"])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                st.write("Recent Authentication Attempts:")
                st.dataframe(df.tail(20))
                # Plot success/failure counts
                status_counts = df['Status'].value_counts()
                fig, ax = plt.subplots()
                status_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
                ax.set_ylabel('Count')
                ax.set_title('Authentication Success vs Failure')
                st.pyplot(fig)
                # Per-user stats
                user_counts = df['Username'].value_counts()
                st.write("Authentication Attempts per User:")
                st.bar_chart(user_counts)
            else:
                st.info("No authentication attempts logged yet.")
        else:
            st.info("No authentication log file found.")

    # --- PDF Export Tab ---
    with tab2:
        st.subheader("Export Logs and Results as PDF")
        from fpdf import FPDF
        def export_pdf(log_file, result_file):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Authentication Log", ln=True, align='C')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    for line in f:
                        pdf.cell(200, 8, txt=line.strip(), ln=True)
            else:
                pdf.cell(200, 8, txt="No log file found.", ln=True)
            pdf.add_page()
            pdf.cell(200, 10, txt="CBT Results", ln=True, align='C')
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    for line in f:
                        pdf.cell(200, 8, txt=line.strip(), ln=True)
            else:
                pdf.cell(200, 8, txt="No results file found.", ln=True)
            return pdf.output(dest='S').encode('latin1')
        if st.button("Export All as PDF"):
            pdf_bytes = export_pdf(LOG_FILE, 'cbt_results.txt')
            st.download_button("Download PDF", pdf_bytes, file_name="cbt_report.pdf")

    # --- User Management Tab ---
    with tab3:
        st.subheader("Registered Users")
        user_files = [f for f in os.listdir(USER_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if user_files:
            for uf in user_files:
                st.write(uf)
                st.image(os.path.join(USER_FOLDER, uf), width=80)
            st.markdown("---")
            st.subheader("Delete User")
            del_user = st.selectbox("Select user to delete", [os.path.splitext(f)[0] for f in user_files])
            admin_pin = st.text_input("Enter Admin PIN to confirm", type="password", key="del_pin")
            if st.button("Delete User"):
                if admin_pin == "123456":  # Replace with secure PIN logic
                    user_img = [f for f in user_files if os.path.splitext(f)[0] == del_user][0]
                    os.remove(os.path.join(USER_FOLDER, user_img))
                    st.success(f"User '{del_user}' deleted.")
                    st.rerun()
                else:
                    st.error("Incorrect Admin PIN.")
        else:
            st.info("No registered users found.")

    # --- Time Limits Tab ---
    with tab4:
        st.subheader("Per-User Time Limit Management")
        for uf in user_files:
            uname = os.path.splitext(uf)[0]
            current_time = get_user_time_limit(uname)
            new_time = st.number_input(f"Time limit for {uname} (seconds)", min_value=30, max_value=3600, value=current_time, step=10, key=f"time_{uname}_tab")
            if st.button(f"Update Time for {uname}", key=f"btn_time_{uname}"):
                set_user_time_limit(uname, new_time)
                st.success(f"Time limit for {uname} set to {new_time} seconds.")

    # --- Manual Reset Tab ---
    with tab5:
        st.subheader("Manual Reset User Attempt")
        reset_user = st.text_input("Enter username to reset CBT attempt", key="reset_user_tab")
        if st.button("Reset Attempt", key="reset_btn_tab"):
            reset_user_attempt(reset_user)
            st.success(f"CBT attempt for '{reset_user}' has been reset.")

    # --- Exam Questions Tab ---
    with tab6:
        st.subheader("Manage CBT Questions")
        QUESTIONS_FILE = 'cbt_questions.json'
        # Load questions
        if os.path.exists(QUESTIONS_FILE):
            with open(QUESTIONS_FILE, 'r') as f:
                questions = json.load(f)
        else:
            questions = []
        # Display questions
        if questions:
            for idx, q in enumerate(questions):
                st.markdown(f"**Q{idx+1}: {q['question']}**")
                for i, opt in enumerate(q['options']):
                    st.write(f"{chr(65+i)}. {opt}")
                st.write(f"Correct Answer: {chr(65+q['answer'])}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Edit Q{idx+1}", key=f"editq{idx}"):
                        st.session_state['edit_q_idx'] = idx
                with col2:
                    if st.button(f"Delete Q{idx+1}", key=f"delq{idx}"):
                        questions.pop(idx)
                        with open(QUESTIONS_FILE, 'w') as f:
                            json.dump(questions, f, indent=2)
                        st.success("Question deleted.")
                        st.rerun()
        else:
            st.info("No questions set yet.")
        st.markdown("---")
        # Add/Edit question
        if 'edit_q_idx' in st.session_state:
            edit_idx = st.session_state['edit_q_idx']
            q = questions[edit_idx]
            q_text = st.text_input("Edit Question", value=q['question'], key="edit_q_text")
            options = [st.text_input(f"Option {chr(65+i)}", value=opt, key=f"edit_opt_{i}") for i, opt in enumerate(q['options'])]
            answer = st.selectbox("Correct Answer", options=[chr(65+i) for i in range(len(options))], index=q['answer'], key="edit_ans")
            if st.button("Save Changes", key="save_edit_q"):
                questions[edit_idx] = {
                    'question': q_text,
                    'options': options,
                    'answer': [chr(65+i) for i in range(len(options))].index(answer)
                }
                with open(QUESTIONS_FILE, 'w') as f:
                    json.dump(questions, f, indent=2)
                st.success("Question updated.")
                del st.session_state['edit_q_idx']
                st.rerun()
            if st.button("Cancel Edit", key="cancel_edit_q"):
                del st.session_state['edit_q_idx']
                st.rerun()
        else:
            st.markdown("### Add New Question")
            new_q = st.text_input("Question Text", key="new_q_text")
            new_opts = [st.text_input(f"Option {chr(65+i)}", key=f"new_opt_{i}") for i in range(4)]
            new_ans = st.selectbox("Correct Answer", options=[chr(65+i) for i in range(4)], key="new_ans")
            if st.button("Add Question", key="add_q"):
                if new_q and all(new_opts):
                    questions.append({
                        'question': new_q,
                        'options': new_opts,
                        'answer': [chr(65+i) for i in range(4)].index(new_ans)
                    })
                    with open(QUESTIONS_FILE, 'w') as f:
                        json.dump(questions, f, indent=2)
                    st.success("Question added.")
                    st.rerun()
                else:
                    st.error("Please fill in all fields.")

    # --- Results Tab ---
    with tab7:
        st.subheader("Student Results & Report Card")
        import pandas as pd
        from fpdf import FPDF
        results_file = 'cbt_results.txt'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            # Parse results: '2024-06-01 12:00:00 - user1 - Score: 8/10'
            data = []
            for l in lines:
                try:
                    dt, user, score = l.split(' - ')
                    score_val = score.split(': ')[1]
                    data.append({'Timestamp': dt, 'Username': user, 'Score': score_val})
                except Exception:
                    continue
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df)
                st.markdown('---')
                st.subheader('Generate Report Card')
                users = df['Username'].unique().tolist()
                selected_user = st.selectbox('Select student', users)
                user_df = df[df['Username'] == selected_user]
                if st.button('Generate Report Card PDF'):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 16)
                    pdf.cell(200, 10, f"Report Card for {selected_user}", ln=True, align='C')
                    pdf.set_font('Arial', '', 12)
                    for idx, row in user_df.iterrows():
                        pdf.cell(200, 10, f"{row['Timestamp']} - Score: {row['Score']}", ln=True)
                    pdf_bytes = pdf.output(dest='S').encode('latin1')
                    st.download_button(f"Download {selected_user} Report Card", pdf_bytes, file_name=f"{selected_user}_report_card.pdf")
            else:
                st.info('No results found.')
        else:
            st.info('No results file found.')

    if st.button("Logout Admin"):
        st.session_state['admin_authenticated'] = False
        st.rerun()

# --- Welcome Page ---
if choice == "Welcome":
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image('unilorin.png', width=120)
    st.markdown("""
    <div style='text-align:center;'>
        <h1 style='color:#007bff;margin-bottom:0.2em;'>University of Ilorin CBT Portal with Facial Recognition</h1>
        <h3 style='color:#222;margin-top:0;'>Empowering Secure, Modern Assessment</h3>
        <p style='font-size:1.1em;color:#555;margin-bottom:20px;'>
            Experience secure, user-friendly authentication and exam access. Developed by ADETOLA JOSHUA YINKA
UIL/PG2023/1826 using advanced face recognition technology.<br>
            <b>University of Ilorin, Ilorin, Nigeria</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# --- Register Page ---
elif choice == "Register":
    st.title("User Registration")
    username = st.text_input("Username")
    captured_image = st.camera_input("Capture your face using webcam")
    if st.button("Register"):
        if not username:
            st.error("Username is required.")
        elif captured_image is None:
            st.error("Face capture is required.")
        else:
            image = Image.open(captured_image)
            image = image.convert('RGB')
            emb = get_face_embedding(image, mtcnn, resnet)
            if emb is not None:
                save_user_image(username, image)
                log_attempt(username, "SUCCESS")
                st.success("Registration successful! You can now authenticate to start your CBT.")
            else:
                st.error("Face embedding failed. Ensure your face is clearly visible in the capture.")

# --- Authenticate & Start CBT Page ---
elif choice == "Authenticate & Start CBT":
    st.title("Authenticate & Start Your CBT")
    username = st.text_input("Username")
    captured_image = st.camera_input("Capture your face for authentication")
    if st.button("Authenticate & Start CBT"):
        if not username:
            st.error("Username is required.")
        elif captured_image is None:
            st.error("Face capture is required.")
        else:
            user_img_path = os.path.join(USER_FOLDER, f"{username}.jpg")
            if os.path.exists(user_img_path):
                stored_image = Image.open(user_img_path)
                stored_emb = get_face_embedding(stored_image, mtcnn, resnet)
                live_image = Image.open(captured_image)
                live_emb = get_face_embedding(live_image, mtcnn, resnet)
                if stored_emb is not None and live_emb is not None:
                    similarity = np.dot(stored_emb, live_emb.T) / (np.linalg.norm(stored_emb) * np.linalg.norm(live_emb))
                    if similarity >= SIMILARITY_THRESHOLD:
                        log_attempt(username, "SUCCESS")
                        st.success("Authentication successful! Redirecting to exam...")
                        st.session_state['authenticated_user'] = username
                        st.session_state['exam_started'] = False
                        st.session_state['exam_answers'] = {}
                        st.session_state['exam_current_q'] = 0
                        st.session_state['exam_submitted'] = False
                        st.experimental_set_query_params(page="Take Exam")
                        st.rerun()
                    else:
                        log_attempt(username, "FAILED")
                        st.error("Face does not match. Authentication failed.")
                else:
                    st.error("Face embedding failed. Try again.")
            else:
                st.error("User not found. Please register first.")

# --- Take Exam Page ---
elif choice == "Take Exam":
    if 'authenticated_user' not in st.session_state:
        st.warning("You must authenticate first.")
        st.stop()
    username = st.session_state['authenticated_user']
    QUESTIONS_FILE = 'cbt_questions.json'
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, 'r') as f:
            questions = json.load(f)
    else:
        questions = CBT_QUESTIONS
    total_questions = len(questions)
    if 'exam_current_q' not in st.session_state:
        st.session_state['exam_current_q'] = 0
    if 'exam_answers' not in st.session_state:
        st.session_state['exam_answers'] = {}
    if 'exam_submitted' not in st.session_state:
        st.session_state['exam_submitted'] = False
    q_idx = st.session_state['exam_current_q']
    if not st.session_state['exam_submitted']:
        st.header(f"CBT Exam - Question {q_idx+1} of {total_questions}")
        q = questions[q_idx]
        answer = st.radio("Select your answer:", q['options'], key=f"exam_q_{q_idx}",
                         index=q['options'].index(st.session_state['exam_answers'].get(q_idx, q['options'][0])) if q_idx in st.session_state['exam_answers'] else 0)
        st.session_state['exam_answers'][q_idx] = answer
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            if st.button("Previous") and q_idx > 0:
                st.session_state['exam_current_q'] -= 1
                st.rerun()
        with col2:
            if st.button("Next") and q_idx < total_questions-1:
                st.session_state['exam_current_q'] += 1
                st.rerun()
        with col3:
            if st.button("Finish and Submit"):
                # Grade exam
                score = 0
                summary = []
                for idx, q in enumerate(questions):
                    correct = q['options'][q['answer']]
                    user_ans = st.session_state['exam_answers'].get(idx, None)
                    is_correct = user_ans == correct
                    summary.append({
                        'question': q['question'],
                        'your_answer': user_ans,
                        'correct_answer': correct,
                        'is_correct': is_correct
                    })
                    if is_correct:
                        score += 1
                with open('cbt_results.txt', 'a') as f:
                    f.write(f"{datetime.now()} - {username} - Score: {score}/{total_questions}\n")
                st.session_state['exam_submitted'] = True
                st.session_state['exam_score'] = score
                st.session_state['exam_summary'] = summary
                st.success('You have successfully submitted your exam!')
                st.rerun()
    else:
        score = st.session_state['exam_score']
        summary = st.session_state['exam_summary']
        st.success(f"Exam submitted! Your score: {score} / {total_questions}")
        st.markdown('---')
        st.subheader('Exam Summary')
        for idx, item in enumerate(summary):
            st.write(f"Q{idx+1}: {item['question']}")
            st.write(f"Your answer: {item['your_answer']}")
            st.write(f"Correct answer: {item['correct_answer']}")
            if item['is_correct']:
                st.success("Correct!")
            else:
                st.error("Incorrect.")
            st.markdown('---')
