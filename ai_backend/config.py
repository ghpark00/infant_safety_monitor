import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

load_dotenv(PROJECT_ROOT / ".env")

# --- 비디오/넘어짐 감지 설정 ---
# --- 모델 경로 설정 ---
# 실제 모델 파일 이름과 경로에 맞게 수정.
HUMAN_MODEL_PATH = str(MODELS_DIR / "yolo11m.pt")
FALL_MODEL_PATH = str(MODELS_DIR / "best.pt")
YOLO_CFG_PATH = str(MODELS_DIR / "yolo-tiny-onecls" / "yolov3-tiny-onecls.cfg")
YOLO_WEIGHT_PATH = str(MODELS_DIR / "yolo-tiny-onecls" / "best-model.pth")
TSSTG_WEIGHT_PATH = str(MODELS_DIR / "TSSTG" / "tsstg-model.pth")

# --- 추론 설정 ---
HUMAN_CONF_THRESHOLD = 0.4  # 사람 탐지 최소 신뢰도
FALL_CONF_THRESHOLD = 0.3   # 넘어짐 탐지 최소 신뢰도
FALL_IOU_THRESHOLD = 0.2    # 넘어짐 탐지 IOU 임계값
PERSON_CLASS_NAME = 'person' # 사람 클래스 이름 (모델에 따라 다를 수 있음)
FALL_CLASS_NAME = 'Fall-Detected'     # 넘어짐 클래스 이름 (모델에 따라 다를 수 있음, 예시)
MAX_PERSON_COUNT = 3  # 최대 허용 인원 수

# --- 알림 설정 ---
# 받는 사람 주소. 실제 스프링부트 서버의 엔드포인트 URL로 변경하면 될듯?.
# 파이썬의 httpx 라이브러리를 사용하여 "POST"라는 방식(데이터를 서버로 보낼 때 주로 사용)으로 요청을 실제로 발송하는 코드가 필요
# httpx.post() 함수 (데이터 포장 및 발송): yolo_inference.py의 send_fall_notification_with_image 함수 안에 있는 다음 코드가 핵심임.
# response = await async_client.post(SPRING_BOOT_FALL_URL, files=files, data=data)
# Spring Boot 서버는 해당 URL(http://localhost:8080/api/notify/fall)에서 multipart/form-data 형식의 POST 요청이 올 것을 미리 알고 준비하고 있어야 함.
SPRING_BOOT_FALL_URL = os.getenv("SPRING_BOOT_FALL_URL", "http://localhost:8080/api/notify/fall_cry_file") # 환경 변수 또는 기본값 사용
SPRING_BOOT_OVERCROWD_URL = os.getenv("SPRING_BOOT_OVERCROWD_URL", "http://localhost:8080/api/alarms/overcrowd")

NOTIFICATION_COOLDOWN_SECONDS = 10 # 알림 재전송 쿨다운 시간 (초)
# --- 웹캠 설정 ---
WEBCAM_INDEX = 0 # 사용할 웹캠 인덱스 (0: 기본 웹캠)

# --- 임시 파일 설정 (추가) ---
TEMP_IMAGE_FOLDER = str(PROJECT_ROOT / "temp_images")
TEMP_VIDEO_FOLDER = str(PROJECT_ROOT / "temp_videos")

#===================================================================================================================================================================================================================
#===================================================================================================================================================================================================================

# --- 오디오/울음소리 감지 설정 (추가) ---
# --- 모델 경로 설정 ---
CRY_MODEL_PATH = str(MODELS_DIR / "my_model.h5")
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1" # YAMNet 경로

# --- 추론 설정 ---
AUDIO_SAMPLE_RATE = 16000  # YAMNet 요구 샘플 속도 (Hz)
AUDIO_CHUNK_DURATION_SECONDS = 5  # 오디오 처리 간격 (초)
AUDIO_DEVICE_INDEX = None  # 사용할 마이크 장치 인덱스 (None: 시스템 기본값)
CRY_CONF_THRESHOLD = 0.6  # 울음소리 판단 임계값

# --- 알림 설정 ---
SPRING_BOOT_CRY_URL = os.getenv("SPRING_BOOT_CRY_URL", "http://localhost:8080/api/notify/cry") # 울음소리 알림 URL
CRY_NOTIFICATION_COOLDOWN_SECONDS = 15 # 울음소리 알림 쿨다운

# --- 임시 파일 설정 (추가) ---
TEMP_AUDIO_FOLDER = str(PROJECT_ROOT / "temp_audio")
