import cv2
import mediapipe as mp
import pyautogui
import time

print("Lütfen bir kontrol yöntemi seçin:")
print("1 - Kafa (burun ile kontrol)")
print("2 - El (işaret parmağı ile kontrol)")
print("3 - Yüz merkezi (yüzün yönüne göre kontrol)")
choice = input("Seçiminiz (1/2/3): ")

CONTROL_METHOD = {"1": "nose", "2": "hand", "3": "face_direction"}.get(choice, "nose")


cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
cooldown = 0.4
last_press_time = 0
last_direction = None
reference_point = None
THRESHOLD = 20           
SMOOTHING = 0.2     


mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

face_mesh = mp_face.FaceMesh() if CONTROL_METHOD == "nose" else None
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6) if CONTROL_METHOD == "hand" else None
face_detector = mp_face_detection.FaceDetection() if CONTROL_METHOD == "face_direction" else None

def detect_direction_delta(current, reference):
    dx = current[0] - reference[0]
    dy = current[1] - reference[1]
    if abs(dx) > abs(dy):
        if dx > THRESHOLD:
            return "right"
        elif dx < -THRESHOLD:
            return "left"
    else:
        if dy > THRESHOLD:
            return "down"
        elif dy < -THRESHOLD:
            return "up"
    return None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame.shape

    current_x, current_y = None, None

    if CONTROL_METHOD == "nose":
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark[1]
            current_x, current_y = int(lm.x * frame_w), int(lm.y * frame_h)
            cv2.circle(frame, (current_x, current_y), 6, (0, 0, 255), -1)

    elif CONTROL_METHOD == "hand":
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark[8]
            current_x, current_y = int(lm.x * frame_w), int(lm.y * frame_h)
            cv2.circle(frame, (current_x, current_y), 6, (255, 0, 0), -1)

    elif CONTROL_METHOD == "face_direction":
        result = face_detector.process(rgb)
        if result.detections:
            bbox = result.detections[0].location_data.relative_bounding_box
            current_x = int((bbox.xmin + bbox.width / 2) * frame_w)
            current_y = int((bbox.ymin + bbox.height / 2) * frame_h)
            cv2.circle(frame, (current_x, current_y), 6, (0, 255, 255), -1)

    if current_x is not None and current_y is not None:
        if reference_point is None:
            reference_point = (current_x, current_y)
        else:
           
            ref_x = int((1 - SMOOTHING) * reference_point[0] + SMOOTHING * current_x)
            ref_y = int((1 - SMOOTHING) * reference_point[1] + SMOOTHING * current_y)
            reference_point = (ref_x, ref_y)

        direction = detect_direction_delta((current_x, current_y), reference_point)
        now = time.time()

        if direction and (direction != last_direction or now - last_press_time > cooldown):
            print(f"Yön: {direction}")
            pyautogui.press(direction)
            last_press_time = now
            last_direction = direction


        if not direction:
            last_direction = None

    cv2.imshow("Daha Hassas Hareket Kontrol", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()