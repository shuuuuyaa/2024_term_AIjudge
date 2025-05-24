import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import os
# Load models
home_base_model = YOLO("weight/best_base_e100.pt")
home_base_model.info()
ball_model = YOLO("weight/best_final.pt")
ball_model.info()

model_path_base = "weight/best_base_e100.pt"
model_path_ball = "weight/best_final.pt"

size_base = os.path.getsize(model_path_base) / (1024 * 1024)  # MB単位
size_ball = os.path.getsize(model_path_ball) / (1024 * 1024)

print(f"Home base model size: {size_base:.2f} MB")
print(f"Ball model size: {size_ball:.2f} MB")

def detect_home_base_vertices(image, masks, class_ids, base_class_id, confidences):
    left_vertex, right_vertex = None, None
    max_confidence = -1

    for mask, class_id, confidence in zip(masks.data, class_ids, confidences):
        if class_id == base_class_id and confidence > max_confidence:
            max_confidence = confidence
            binary_mask = (mask.cpu().numpy() > 0.5).astype(np.uint8)
            binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                left_vertex = (x, y + h)  # Bottom-left
                right_vertex = (x + w, y + h)  # Bottom-right
    return left_vertex, right_vertex, max_confidence

def detect_largest_person(image):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return None, None

        h, w, _ = image.shape
        person_bboxes = [(int(landmark.x * w), int(landmark.y * h)) for landmark in results.pose_landmarks.landmark]

        x_coords = [pt[0] for pt in person_bboxes]
        y_coords = [pt[1] for pt in person_bboxes]
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

        return results.pose_landmarks, bbox

def draw_custom_shape(image, right_vertex, left_vertex, avg_knee_coords, avg_shoulder_coords):
    if right_vertex and left_vertex:
        c1 = (right_vertex[0], avg_knee_coords[1])
        c2 = (left_vertex[0], avg_knee_coords[1])
        c3 = (right_vertex[0], avg_shoulder_coords[1])
        c4 = (left_vertex[0], avg_shoulder_coords[1])

        cv2.line(image, c1, c2, (0, 0, 255), 2)
        cv2.line(image, c2, c4, (0, 0, 255), 2)
        cv2.line(image, c4, c3, (0, 0, 255), 2)
        cv2.line(image, c3, c1, (0, 0, 255), 2)

        return image, (c1, c2, c3, c4)
    return image, None

def crop_red_box(image, c1, c2, c3, c4):
    x_min = min(c1[0], c2[0], c3[0], c4[0])
    x_max = max(c1[0], c2[0], c3[0], c4[0])
    y_min = min(c1[1], c2[1], c3[1], c4[1])
    y_max = max(c1[1], c2[1], c3[1], c4[1])

    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

def process_frame(frame):
    results = home_base_model(frame)
    masks = results[0].masks
    if masks is None:
        print("No masks detected. Skipping frame.")
        return frame  # または None を返してスキップしてもOK

    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    base_class_id = 0  # 適切なクラスIDに変更

    left_vertex, right_vertex, max_confidence = detect_home_base_vertices(frame, masks, class_ids, base_class_id, confidences)

    if max_confidence > -1:
        print(f"Detected home base with confidence: {max_confidence}")

    landmarks, bbox = detect_largest_person(frame)

    if landmarks:
        h, w, _ = frame.shape
        mp_pose = mp.solutions.pose
        r_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        l_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        r_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        l_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

        avg_knee_coords = (
            int((r_knee.x + l_knee.x) / 2 * w),
            int((r_knee.y + l_knee.y) / 2 * h)
        )

        avg_shoulder_hip_mid_coords = (
            int(((r_shoulder.x + l_shoulder.x) / 2 + (r_hip.x + l_hip.x) / 2 ) / 2 * w),
            int(((r_shoulder.y + l_shoulder.y) / 2 + (r_hip.y + l_hip.y) / 2 ) / 2 * h)
        )

        output_image, red_box_coords = draw_custom_shape(frame, right_vertex, left_vertex, avg_knee_coords, avg_shoulder_hip_mid_coords)

        if red_box_coords:
            cropped_frame = crop_red_box(frame, *red_box_coords)
            resized_frame = cv2.resize(cropped_frame, (cropped_frame.shape[1] * 3, cropped_frame.shape[0] * 3))
            ball_results = ball_model(resized_frame)

            if ball_results and ball_results[0].boxes and ball_results[0].boxes.conf.numel() > 0:
                boxes = ball_results[0].boxes
                confs = boxes.conf.cpu().numpy()
                best_box_idx = confs.argmax()
                best_box = boxes[best_box_idx]

                x_min, y_min, x_max, y_max = best_box.xyxy[0].cpu().tolist()
                ball_width = x_max - x_min

                print("Highest confidence detection:")
                print(f"Label: {best_box.cls.item()}, Confidence: {best_box.conf.item():.2f}, Coordinates: {best_box.xyxy.cpu().numpy()}")
                print(f"Width of the highest confidence object: {ball_width}")

                h, w = resized_frame.shape[:2]
                if w // ball_width >= 6 and w // ball_width <= 7:
                    print("strike")
                else:
                    print("ball")
            else:
                print("No ball detected.")

    return frame

def main():
    video_path = 'test/judge_test.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        out = process_frame(frame)
        cv2.imwrite("debug_output.jpg", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    print(f"全体の推論時間: {end_time - start_time:.2f} 秒")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()