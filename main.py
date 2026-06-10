import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
# במחשב מקומי, הדרך הרגילה עובדת מצוין!
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 82, 312, 308, 317, 87]

def get_coords(landmarks, indices, img_w, img_h):
    return np.array([[landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h] for i in indices])

ear_values = []
mar_values = []

# שם הסרטון (חייב להיות באותה תיקייה כמו הקובץ main.py)
video_name = 'vid1.mp4'
cap = cv2.VideoCapture(video_name)

if not cap.isOpened():
    print(f"Error: Could not open {video_name}. Make sure it's in the correct folder.")
    exit()

print("Processing video frames... Please wait.")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 
        
    img_h, img_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            right_eye = get_coords(face_landmarks, RIGHT_EYE_INDICES, img_w, img_h)
            left_eye = get_coords(face_landmarks, LEFT_EYE_INDICES, img_w, img_h)
            mouth = get_coords(face_landmarks, MOUTH_INDICES, img_w, img_h)
            
            def calc_ear(eye):
                v1 = np.linalg.norm(eye[1] - eye[5])
                v2 = np.linalg.norm(eye[2] - eye[4])
                h = np.linalg.norm(eye[0] - eye[3])
                return (v1 + v2) / (2.0 * h)
                
            def calc_mar(m):
                v1 = np.linalg.norm(m[1] - m[5])
                v2 = np.linalg.norm(m[2] - m[4])
                h = np.linalg.norm(m[0] - m[3])
                return (v1 + v2) / (2.0 * h)
            
            ear = (calc_ear(right_eye) + calc_ear(left_eye)) / 2.0
            mar = calc_mar(mouth)
            
            ear_values.append(ear)
            mar_values.append(mar)
    else:
        ear_values.append(0)
        mar_values.append(0)
    
    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
print("Processing complete! Plotting graphs...")

# ציור הגרפים נפתח בחלון נפרד במחשב שלך
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(ear_values, label='EAR (Eye Aspect Ratio)', color='blue', linewidth=2)
plt.title('Eye State Over Time', fontsize=14)
plt.ylabel('EAR', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(mar_values, label='MAR (Mouth Aspect Ratio)', color='red', linewidth=2)
plt.title('Mouth State Over Time (Yawning Detection)', fontsize=14)
plt.xlabel('Frames', fontsize=12)
plt.ylabel('MAR', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()