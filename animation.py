import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def animate_face(image, frames=30):
    height, width, _ = image.shape
    video_frames = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_image)

        if not result.multi_face_landmarks:
            raise Exception("No face detected!")

        face_landmarks = result.multi_face_landmarks[0]

        # Coordinates for eyes based on Mediapipe face mesh indexes
        left_eye_idxs = [33, 160, 158, 133, 153, 144]
        right_eye_idxs = [362, 385, 387, 263, 373, 380]

        for i in range(frames):
            frame = image.copy()
            for eye_landmarks in [left_eye_idxs, right_eye_idxs]:
                points = []
                for idx in eye_landmarks:
                    x = int(face_landmarks.landmark[idx].x * width)
                    y = int(face_landmarks.landmark[idx].y * height)
                    points.append([x, y])
                points = np.array(points, dtype=np.int32)

                if i % 10 < 5:  # blink animation
                    cv2.fillPoly(frame, [points], color=(0, 0, 0))

            video_frames.append(frame)

    return video_frames
