import os

import utils
import shared

import cv2
import mediapipe as mp
import threading
import math
import time

def face_detect_loop():
    crop = True

    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    zoom_factor = 1.7
    process_every_n_frames = 2  # Process 1 frame every 2 frames
    frame_count2 = 0
    target_frame_width = 320
    target_frame_height = 240

    forehead = 10
    chin = 175
    l_cheek = 234
    c_cheek = 5
    r_cheek = 454
    l_iris = 468
    r_iris = 473
    l_upper_lid = 159
    r_upper_lid = 386
    l_lower_lid = 145
    r_lower_lid = 374
    l_outer_edge = 246
    l_inner_edge = 173
    r_outer_edge = 466
    r_inner_edge = 398
    upper_lip = 13
    lower_lip = 14
    l_edge_lip = 61
    r_edge_lip = 291
    l_brow = 55
    r_brow = 285

    key_landmarks = [forehead, chin, l_cheek, c_cheek, r_cheek, l_iris, r_iris, l_upper_lid, r_upper_lid, l_lower_lid, r_lower_lid, l_outer_edge, l_inner_edge, r_outer_edge, r_inner_edge, upper_lip, lower_lip, l_edge_lip, r_edge_lip, l_brow, r_brow]

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    frame_count = 0
    start_time = time.time()
    fps = 0

    while shared.running:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break
        
        frame = cv2.resize(frame, (target_frame_width, target_frame_height))

        # Only process every N frames
        if frame_count2 % process_every_n_frames == 0:
            frame_count += 1
            elapsed_time = time.time() - start_time

            if elapsed_time >= 1.0:  # Every 1 second
                fps = frame_count / elapsed_time
                # print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
            
            # Calculate cropping coordinates
            if crop:
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, (h // 2) - 30
                radius_x, radius_y = int(w / (2 * zoom_factor)), int(h / (2 * zoom_factor))

                min_x, max_x = center_x - radius_x, center_x + radius_x
                min_y, max_y = center_y - radius_y, center_y + radius_y

                # Crop and resize (simulate zoom)
                cropped = frame[min_y:max_y, min_x:max_x]
                frame = cv2.resize(cropped, (w, h))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # mp_drawing.draw_landmarks(
                    #     frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(50, 200, 50), thickness=1)
                    # )

                    shared.delta_y = (math.dist(utils.get_pos(face_landmarks.landmark[chin]), utils.get_pos(face_landmarks.landmark[c_cheek]))) - (math.dist(utils.get_pos(face_landmarks.landmark[c_cheek]), utils.get_pos(face_landmarks.landmark[forehead])))

                    shared.delta_x = ((math.dist(utils.get_pos(face_landmarks.landmark[r_cheek]), utils.get_pos(face_landmarks.landmark[c_cheek]))) - (math.dist(utils.get_pos(face_landmarks.landmark[c_cheek]), utils.get_pos(face_landmarks.landmark[l_cheek])))) / (shared.HEAD_W/2)

                    shared.head_h = math.dist(utils.get_pos(face_landmarks.landmark[chin]), utils.get_pos(face_landmarks.landmark[forehead]))
                    shared.head_w = math.dist(utils.get_pos(face_landmarks.landmark[r_cheek]), utils.get_pos(face_landmarks.landmark[l_cheek]))

                    shared.l_eye_w = math.dist(utils.get_pos(face_landmarks.landmark[l_inner_edge]), utils.get_pos(face_landmarks.landmark[l_outer_edge]))
                    shared.r_eye_w = math.dist(utils.get_pos(face_landmarks.landmark[r_inner_edge]), utils.get_pos(face_landmarks.landmark[r_outer_edge]))

                    shared.l_eye_h = math.dist(utils.get_pos(face_landmarks.landmark[l_lower_lid]), utils.get_pos(face_landmarks.landmark[l_upper_lid]))
                    shared.r_eye_h = math.dist(utils.get_pos(face_landmarks.landmark[r_lower_lid]), utils.get_pos(face_landmarks.landmark[r_upper_lid]))

                    shared.l_eye_pct = (shared.l_eye_h/shared.head_h) * (shared.HEAD_H/shared.EYE_H)
                    shared.r_eye_pct = (shared.r_eye_h/shared.head_h) * (shared.HEAD_H/shared.EYE_H)

                    shared.l_eye_center_x, shared.l_eye_center_y = utils.get_intersection(
                        *utils.get_pos(face_landmarks.landmark[l_outer_edge]),
                        *utils.get_pos(face_landmarks.landmark[l_inner_edge]),
                        *utils.get_pos(face_landmarks.landmark[l_upper_lid]),
                        *utils.get_pos(face_landmarks.landmark[l_lower_lid])
                        )
                    shared.l_eye_center = (shared.l_eye_center_x, shared.l_eye_center_y)
                    
                    shared.r_eye_center_x, shared.r_eye_center_y = utils.get_intersection(
                        *utils.get_pos(face_landmarks.landmark[r_outer_edge]),
                        *utils.get_pos(face_landmarks.landmark[r_inner_edge]),
                        *utils.get_pos(face_landmarks.landmark[r_upper_lid]),
                        *utils.get_pos(face_landmarks.landmark[r_lower_lid])
                        )
                    shared.r_eye_center = (shared.r_eye_center_x, shared.r_eye_center_y)
                    
                    shared.l_iris_x = (
                        (
                            math.dist(utils.get_pos(face_landmarks.landmark[l_outer_edge]),
                                        utils.get_pos(face_landmarks.landmark[l_iris])) -
                            math.dist(utils.get_pos(face_landmarks.landmark[l_inner_edge]),
                                        utils.get_pos(face_landmarks.landmark[l_iris]))
                        )
                        /
                        (shared.l_eye_w / 2)
                    )
                    
                    shared.l_iris_y = (
                        (
                            math.dist(utils.get_pos(face_landmarks.landmark[l_upper_lid]),
                                        utils.get_pos(face_landmarks.landmark[l_iris])) -
                            math.dist(utils.get_pos(face_landmarks.landmark[l_lower_lid]),
                                        utils.get_pos(face_landmarks.landmark[l_iris]))
                        )
                        /
                        (shared.l_eye_h / 2)
                    )
                    
                    shared.r_iris_x = (
                        (
                            math.dist(utils.get_pos(face_landmarks.landmark[r_outer_edge]),
                                        utils.get_pos(face_landmarks.landmark[r_iris])) -
                            math.dist(utils.get_pos(face_landmarks.landmark[r_inner_edge]),
                                        utils.get_pos(face_landmarks.landmark[r_iris]))
                        )
                        /
                        (shared.r_eye_w / 2)
                    )
                    
                    shared.r_iris_y = (
                        (
                            math.dist(utils.get_pos(face_landmarks.landmark[r_upper_lid]),
                                        utils.get_pos(face_landmarks.landmark[r_iris])) -
                            math.dist(utils.get_pos(face_landmarks.landmark[r_lower_lid]),
                                        utils.get_pos(face_landmarks.landmark[r_iris]))
                        )
                        /
                        (shared.r_eye_h / 2)
                    )

                    shared.mouth_h = math.dist(utils.get_pos(face_landmarks.landmark[lower_lip]), utils.get_pos(face_landmarks.landmark[upper_lip]))
                    shared.mouth_pct = (shared.mouth_h/shared.head_h) * (shared.HEAD_H/shared.MOUTH)

                    # DEBUG
                    cv2.putText(frame, f"shared.l_iris_y: {shared.l_iris_y:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.putText(frame, f"shared.l_iris_y: {shared.l_iris_y:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.putText(frame, f"shared.r_iris_y: {shared.r_iris_y:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.putText(frame, f"shared.r_iris_y: {shared.r_iris_y:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    cv2.putText(frame, f"shared.l_eye_pct: {shared.l_eye_pct:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.putText(frame, f"shared.l_eye_pct: {shared.l_eye_pct:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.putText(frame, f"shared.r_eye_pct: {shared.r_eye_pct:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                    cv2.putText(frame, f"shared.r_eye_pct: {shared.r_eye_pct:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    if idx in key_landmarks:
                        cv2.circle(frame, (x,y), 2, (0, 0, 255), -1)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            processed_frame = frame
            cv2.imshow("Face Mesh", processed_frame)
        else:
            cv2.imshow("Face Mesh", processed_frame)
        
        frame_count2 += 1
        

        if cv2.waitKey(1) & 0xFF == 27:
            shared.running = False
            break

def start():
    shared.running = True
    threading.Thread(target=face_detect_loop, daemon=False).start()