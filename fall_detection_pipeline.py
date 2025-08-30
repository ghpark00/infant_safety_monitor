# fall_detection_pipeline.py
import numpy as np
import torch
import cv2

from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG
from Detection.Utils import ResizePadding
from fn import draw_single
from config import MAX_PERSON_COUNT
from Detection.Utils import rescale_boxes # rescale_boxes 함수를 임포트

# --- 전역 객체 ---
detect_model = None
pose_model = None
tracker = None
action_model = None
resize_fn = None


def initialize_models(device):
    """모델 초기화 (FastAPI 앱 시작 시 호출 필요)"""
    global detect_model, pose_model, tracker, action_model, resize_fn

    print("[INFO] TinyYOLOv3_onecls 초기화 중...")
    detect_model = TinyYOLOv3_onecls(384, device=device)
    print("[INFO] SPPE_FastPose 초기화 중...")
    pose_model = SPPE_FastPose('resnet50', 224, 160, device=device)
    print("[INFO] Tracker 초기화 중...")
    tracker = Tracker(max_age=30, n_init=3)
    print("[INFO] TSSTG 초기화 중...")
    action_model = TSSTG()
    resize_fn = ResizePadding(384, 384)

    print("[✔] 모든 모델이 성공적으로 초기화되었습니다.")


def kpt2bbox(kpt, ex=20):
    """관절 좌표로부터 bbox 생성"""
    return np.array([
        kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
        kpt[:, 0].max() + ex, kpt[:, 1].max() + ex
    ])


def process_frame(frame):
    """단일 프레임을 처리하여 낙상 여부 및 사람 수 반환 (좌표계 변환 적용)"""
    global detect_model, pose_model, tracker, action_model, resize_fn

    # 원본 프레임의 높이와 너비 저장
    orig_h, orig_w = frame.shape[:2]
    # 모델 입력 크기 (하드코딩, 필요시 인자로 받도록 수정 가능)
    current_dim = 384

    # 프레임 리사이즈
    resized = resize_fn(frame.copy())
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 1. 사람 탐지
    detected = detect_model.detect(rgb, need_resize=False, expand_bb=10)

    # 2. 이전 프레임의 추적 정보로 Kalman 예측
    tracker.predict()
    for track in tracker.tracks:
        det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
        detected = torch.cat([detected, det], dim=0) if detected is not None else det

    # 3. 관절 예측
    detections = []
    if detected is not None:
        poses = pose_model.predict(rgb, detected[:, :4], detected[:, 4])
        for ps in poses:
            kpts = ps['keypoints'].numpy()
            scores = ps['kp_score'].numpy()
            bbox = kpt2bbox(kpts, ex=0)
            detections.append(Detection(bbox, np.concatenate((kpts, scores), axis=1), scores.mean()))

    # 4. 추적기 업데이트
    tracker.update(detections)

    # 5. 확정된 트랙 기반으로 인원 수 계산 및 표시
    confirmed_tracks = [t for t in tracker.tracks if t.is_confirmed() and t.time_since_update == 0]
    person_count = len(confirmed_tracks)
    limit_exceeded = person_count > MAX_PERSON_COUNT
    text_color = (0, 0, 255) if limit_exceeded else (0, 255, 0)
    cv2.putText(frame, f"Persons: {person_count}/{MAX_PERSON_COUNT}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    fall_detected = False

    for track in tracker.tracks:
        if not track.is_confirmed():
            continue

        bbox_resized = track.to_tlbr()
        kpts_resized = track.keypoints_list[-1]

        # --- ★★★ 좌표를 원본 프레임 크기로 변환 ★★★ ---
        # 1. BBox 좌표 변환 (rescale_boxes 함수 사용)
        #   rescale_boxes는 numpy 배열을 받으므로 변환 필요
        bbox_orig_np = rescale_boxes(np.array([bbox_resized]), current_dim, (orig_h, orig_w))
        bbox_orig = tuple(bbox_orig_np[0].astype(int))

        # 2. Keypoints 좌표 변환 (rescale_boxes 로직을 직접 적용)
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(orig_h, orig_w))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(orig_h, orig_w))
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x
        
        kpts_orig = kpts_resized.copy()
        kpts_orig[:, 0] = ((kpts_resized[:, 0] - pad_x // 2) / unpad_w) * orig_w
        kpts_orig[:, 1] = ((kpts_resized[:, 1] - pad_y // 2) / unpad_h) * orig_h
        
        center_orig = (int((bbox_orig[0] + bbox_orig[2]) / 2), int((bbox_orig[1] + bbox_orig[3]) / 2))
        # --- ★★★ 변환 완료 ★★★ ---

        action = 'Pending...'
        action_name = ''
        color = (0, 255, 0)

        # 6. 낙상 예측
        if len(track.keypoints_list) == 30:
            pts = np.array(track.keypoints_list, dtype=np.float32)
            out = action_model.predict(pts, rgb.shape[:2])
            action_name = action_model.class_names[out[0].argmax()]
            ################
            fall_prob = out[0][action_model.class_names.index('Fall Down')]  # 'Fall Down'의 확률
            ###############
            action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)

            if action_name == 'Fall Down' and fall_prob >= 0.10: # 0.1 ~ 0.3
                fall_detected = True
                color = (0, 0, 255)
            elif action_name == 'Lying Down':
                color = (0, 165, 255)

        # 7. 시각화 (변환된 'orig' 좌표 사용)
        if track.time_since_update == 0:
            draw_single(frame, kpts_orig)
            cv2.rectangle(frame, (bbox_orig[0], bbox_orig[1]), (bbox_orig[2], bbox_orig[3]), color, 2)
            cv2.putText(frame, f"ID {track.track_id}", (center_orig[0], center_orig[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, action, (bbox_orig[0], bbox_orig[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame, fall_detected, person_count
