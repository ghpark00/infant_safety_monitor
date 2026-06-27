from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import cv2
import datetime
import os

from ai_backend.config import TEMP_VIDEO_FOLDER

def init_gdrive():
    gauth = GoogleAuth()

    # 인증 정보가 이미 저장되어 있으면, 저장된 토큰을 사용해서 인증을 건너뛰도록 함
    gauth.LoadCredentialsFile("my_credentials.json")
    
    # 인증 정보가 없거나 유효하지 않으면, 새로 인증을 받음
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()  # 브라우저 열어서 로그인 후 인증
    elif gauth.access_token_expired:
        gauth.Refresh()  # 만료된 토큰을 갱신
    else:
        gauth.Authorize()  # 인증이 완료된 토큰을 사용

    # 인증 정보 저장
    gauth.SaveCredentialsFile("my_credentials.json")

    drive = GoogleDrive(gauth)
    return drive

# init_gdrive()


def save_video(frames, camera_id="CAM01"):
    # === 1. 파일 이름 생성 ===
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H%M")
    filename = f"{timestamp}_{camera_id}.mp4"
    os.makedirs(TEMP_VIDEO_FOLDER, exist_ok=True)
    local_path = os.path.join(TEMP_VIDEO_FOLDER, filename)

    print(f"🎬 영상 저장 경로: {local_path}")
    print(f"🎞️ 저장할 프레임 수: {len(frames)}")
    
    # === 2. 로컬에 영상 저장 ===
    try:
        if not frames: # 프레임 버퍼가 비어있는 경우 처리
            print("❌ 저장할 프레임이 없습니다.")
            return None # 또는 "NO_FRAMES" 같은 특정 문자열 반환
        
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(local_path, fourcc, 30.0, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        print("✅ 로컬 영상 저장 완료")
        # 로컬 저장이 성공하면 local_path를 반환
        return local_path # <--- 이 부분을 추가.
        
    except Exception as e:
        print(f"❌ 영상 저장 실패: {e}")
        return "LOCAL_SAVE_FAILED" # 로컬 저장 실패 시
