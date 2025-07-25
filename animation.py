import cv2
import numpy as np
import face_recognition

def animate_face(img, frames=30):
    height, width, _ = img.shape
    video_frames = []

    # Detect faces
    face_landmarks_list = face_recognition.face_landmarks(img)

    if not face_landmarks_list:
        raise Exception("No face detected!")

    for i in range(frames):
        frame = img.copy()
        for face_landmarks in face_landmarks_list:
            # Animate eyes by drawing blinking effect
            for eye in ['left_eye', 'right_eye']:
                pts = np.array(face_landmarks[eye], np.int32)
                if i % 10 < 5:  # blink every 10 frames
                    cv2.fillPoly(frame, [pts], color=(0, 0, 0))
        video_frames.append(frame)

    return video_frames
