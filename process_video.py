# process_video.py (실시간 확인 기능 추가 버전)

import cv2
import torch
import argparse
from tqdm import tqdm
import time
import asyncio
import os
import datetime
from collections import deque

# --- 오디오 처리를 위한 라이브러리 ---
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    print("오류: moviepy 라이브러리가 설치되지 않았습니다.")
    print("터미널에 'pip install moviepy'를 입력하여 설치해주세요.")
    exit()

# --- 기존 모듈에서 필요 함수 임포트 ---
from fall_detection_pipeline import initialize_models, process_frame
from audio_inference import load_audio_models, predict_cry_from_file, ensure_temp_audio_folder
from yolo_inference import send_overcrowd_notification, send_video_notification

# --- 설정값 임포트 ---
from config import MAX_PERSON_COUNT, TEMP_AUDIO_FOLDER, AUDIO_SAMPLE_RATE

# 버퍼에서 비디오 클립을 저장하는 헬퍼 함수
def save_clip_from_buffer(buffer: deque, fps: int, output_folder="temp_clips"):
    """메모리 버퍼에 저장된 프레임들로 짧은 비디오 클립을 생성"""
    if not buffer:
        return None
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_folder, f"clip_{timestamp}.mp4")
    
    height, width, _ = buffer[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    
    print(f"알림용 비디오 클립 저장 중... 경로: {path}")
    for frame in buffer:
        out.write(frame)
        
    out.release()
    print("클립 저장이 완료되었습니다.")
    return path

def process_video_file(input_path: str, output_path: str, user_id: str, enable_notifications: bool):
    """
    비디오 파일을 처리하여 낙상 및 인원수를 탐지하고,
    조건 충족 시 알림을 보내고, 결과를 비디오 파일로 저장하며,
    처리 과정을 실시간으로 화면에 보여줌줌
    """
    # --- 1. 모델 초기화 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    initialize_models(device)
    if enable_notifications:
        print("\n[알림 모드 활성화] 오디오 모델을 로드합니다.")
        load_audio_models()
        ensure_temp_audio_folder()
        
    print("\n[INFO] 모든 모델이 성공적으로 로드되었습니다.")

    # --- 2. 비디오 파일 열기 ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] 비디오 파일을 열 수 없습니다: {input_path}")
        return

    # --- 3. 비디오 정보 추출 ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # --- 4. 오디오 추출 (알림 기능 활성화 시) ---
    temp_audio_path = None
    if enable_notifications:
        try:
            print("\n영상에서 오디오를 추출하는 중입니다... (시간이 걸릴 수 있습니다)")
            video_clip = VideoFileClip(input_path)
            temp_audio_path = os.path.join(TEMP_AUDIO_FOLDER, f"temp_audio_{os.path.basename(input_path)}.wav")
            video_clip.audio.write_audiofile(temp_audio_path, fps=AUDIO_SAMPLE_RATE, logger=None) # logger=None 추가
            print(f"오디오가 임시 파일로 저장되었습니다: {temp_audio_path}")
        except Exception as e:
            print(f"[경고] 오디오 추출에 실패했습니다: {e}. 울음소리 감지 기능이 비활성화됩니다.")
            temp_audio_path = None

    # --- 5. 알림 관련 변수 및 버퍼 초기화 ---
    frame_buffer = deque(maxlen=int(fps * 3))
    last_overcrowd_time = 0
    last_fall_notify_time = 0
    NOTIFICATION_COOLDOWN = 30

    print("-" * 50)
    print(f"비디오 처리 시작: {input_path}")
    print(f"총 프레임 수: {total_frames}, FPS: {fps}")
    if enable_notifications:
        print(f"알림 받을 사용자 ID: {user_id}")
    print(f"결과 저장 경로: {output_path}")
    print("실시간 확인 창에서 'q' 키를 누르면 종료됩니다.")
    print("-" * 50)

    # --- 6. 프레임 단위 처리 ---
    for _ in tqdm(range(total_frames), desc="프레임 처리 중"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame.copy())
        
        processed_frame, fall_detected, person_count = process_frame(frame)

        # --- 알림 로직 (알림 기능이 활성화된 경우에만 실행) ---
        if enable_notifications and user_id:
            current_time = time.time()

            if person_count > MAX_PERSON_COUNT:
                if (current_time - last_overcrowd_time) > NOTIFICATION_COOLDOWN:
                    print(f"\n[!] 인원 초과 감지! (현재: {person_count}명) 알림을 전송합니다.")
                    asyncio.run(send_overcrowd_notification(person_count, user_id))
                    last_overcrowd_time = current_time

            if fall_detected:
                if (current_time - last_fall_notify_time) > NOTIFICATION_COOLDOWN:
                    print(f"\n[!] 낙상 감지! 오디오 분석을 시작합니다.")
                    is_crying = False
                    if temp_audio_path:
                        is_crying, prob = predict_cry_from_file(temp_audio_path)
                        if is_crying:
                            print(f"[!] 울음소리 감지됨! (확률: {prob:.2f})")
                        else:
                            print("[-] 울음소리가 감지되지 않았습니다.")
                    else:
                        print("[경고] 추출된 오디오 파일이 없어 울음소리 분석을 건너뜁니다.")

                    if is_crying:
                        print(">>> 낙상과 울음소리가 모두 감지되어 비디오 클립과 함께 알림을 보냅니다.")
                        clip_path = save_clip_from_buffer(frame_buffer, fps)
                        if clip_path:
                            asyncio.run(send_video_notification(clip_path, user_id))
                            try:
                                os.remove(clip_path)
                            except OSError as e:
                                print(f"임시 클립 삭제 실패: {e}")
                        
                        last_fall_notify_time = current_time
        
        # 결과 비디오 파일에 저장
        out_video.write(processed_frame)

        # ★★★ 실시간으로 화면에 보여주기 ★★★
        cv2.imshow('Real-time Detection', processed_frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n사용자가 'q'를 눌러 처리를 중단합니다.")
            break

    # --- 7. 자원 해제 ---
    print("\n" + "=" * 50)
    print(f"비디오 처리가 완료되었습니다. 결과가 '{output_path}'에 저장되었습니다.")
    print("=" * 50)
    
    cap.release()
    out_video.release()
    cv2.destroyAllWindows() # ★★★ 열려있는 모든 OpenCV 창을 닫음 ★★★

    if temp_audio_path and os.path.exists(temp_audio_path):
        try:
            os.remove(temp_audio_path)
            print(f"임시 오디오 파일 삭제 완료: {temp_audio_path}")
        except OSError as e:
            print(f"임시 오디오 파일 삭제 실패: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MP4 비디오 파일에서 낙상 및 인원을 탐지하고 알림을 보냅니다.")
    
    parser.add_argument('--input', type=str, required=True, help="분석할 원본 MP4 비디오 파일 경로")
    parser.add_argument('--output', type=str, default='output.mp4', help="처리된 결과 비디오를 저장할 파일 경로 (기본값: output.mp4)")
    parser.add_argument('--userId', type=str, help="알림을 수신할 사용자의 ID")
    parser.add_argument('--notify', action='store_true', help="이 플래그를 설정하면 인원 초과, 낙상+울음 감지 시 알림을 활성화합니다.")

    args = parser.parse_args()

    if args.notify and not args.userId:
        parser.error("--notify 플래그를 사용하려면 --userId 인자가 반드시 필요합니다.")

    process_video_file(args.input, args.output, args.userId, args.notify)