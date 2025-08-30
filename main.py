# main.py
import cv2
import asyncio
import os
import datetime
import sounddevice as sd
import torch
import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from scipy.io.wavfile import write as write_wav
from collections import deque
from store_video import save_video
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Union

# 최대 300프레임 (30fps 기준 3초) 저장하는 버퍼
MAX_BUFFER_SIZE=90
frame_buffer = deque(maxlen=MAX_BUFFER_SIZE)
# 임시 비디오 파일을 저장할 디렉토리 경로 설정
TEMP_VIDEO_FOLDER = "path_to_your_temp_video_folder"  # 예시: "temp_videos"

# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(TEMP_VIDEO_FOLDER):
    os.makedirs(TEMP_VIDEO_FOLDER)

# 추론 모듈 임포트
from fall_detection_pipeline import process_frame

# 기존 임포트들 유지
from yolo_inference import (
    send_video_notification, send_overcrowd_notification
)
from audio_inference import (
    load_audio_models, predict_cry_from_file,
    ensure_temp_audio_folder
)
from config import (
    TEMP_IMAGE_FOLDER, AUDIO_SAMPLE_RATE, # async_client,
    AUDIO_CHUNK_DURATION_SECONDS, AUDIO_DEVICE_INDEX, TEMP_AUDIO_FOLDER, MAX_PERSON_COUNT
)

# 모델 클래스 정의
class CameraControlRequest(BaseModel):
    cctvAddress: Union[str, int]  # 문자열 또는 숫자 허용
    userId: str
    action: str  # 'START' or 'STOP'

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 전역 상태 변수
class AppState:
    def __init__(self):
        self.video_capture = None
        self.audio_monitoring_task = None
        self.userId = None
        self.is_detection_active = False
        self.recent_cry_detected = False
        self.recent_fall_detected = False
        self.last_overcrowd_time=0
        self.last_fall_time = 0
        self.frame_buffer = []  # 최근 3초 동안 프레임 저장
        self.last_fall_t="00:00" #테스트용 시간 값
        
        # New attributes for fall detection audio cooldown
        self.cooldown_until_next_audio_check = 0.0  # Initialize to allow first check immediately
        self.MIN_AUDIO_CHECK_INTERVAL = 30  # 30 seconds cooldown

app_state = AppState()

# --- FastAPI 애플리케이션 수명 관리 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 실행될 코드
    # 임시 폴더 생성 확인/생성
    if not ensure_temp_audio_folder(): print("중요: 임시 오디오 폴더를 확보하지 못했습니다.")
    
    # 추론 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스 : {device}")
    from fall_detection_pipeline import initialize_models
    initialize_models(device=device)  # or 'cpu'
    
    # 오디오 모델 로드
    load_audio_models()
    print("서버가 실행중입니다...")
    # 웹캠은 시작 시 열지 않고, 앱에서 시작 요청 시 열도록 변경
    
    yield
    
    # 애플리케이션 종료
    print("애플리케이션 종료...")
    # 백그라운드 작업 취소
    if app_state.audio_monitoring_task and not app_state.audio_monitoring_task.done():
        print("오디오 모니터링 작업을 취소합니다...")
        app_state.audio_monitoring_task.cancel()
        try:
            await app_state.audio_monitoring_task
        except asyncio.CancelledError:
            print("오디오 모니터링 작업이 취소되었습니다.")

    # 웹캠 자원 해제
    if app_state.video_capture and app_state.video_capture.isOpened():
        app_state.video_capture.release()
        print("웹캠 자원 해제.")
        
        
app = FastAPI(lifespan=lifespan)


# 웹캠 제어 엔드포인트
@app.post("/control_camera")
async def control_camera(request: CameraControlRequest):
    if request.action == "START":
        # 웹캠 시작 로직
        try:
            # 기존에 열려 있는 웹캠이 있으면 닫기
            if app_state.video_capture and app_state.video_capture.isOpened():
                app_state.video_capture.release()
            
            # 웹캠 URL이 숫자면 정수로 변환
            camera_index = int(request.cctvAddress) if str(request.cctvAddress).isdigit() else request.cctvAddress
            
            # 새로운 웹캠 URL로 웹캠 열기
            app_state.video_capture = cv2.VideoCapture(camera_index)
            if not app_state.video_capture.isOpened():
                return JSONResponse(
                    status_code=400,
                    content={"message": f"웹캠을 열 수 없습니다: {request.cctvAddress}"}
                )
            
            app_state.userId = request.userId
            app_state.is_detection_active = True
            app_state.cooldown_until_next_audio_check = 0.0
            
            return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
        
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"message": f"웹캠 시작 중 오류 발생: {str(e)}"}
            )
    
    elif request.action == "STOP":
        # 웹캠 종료 로직
        app_state.is_detection_active = False
        if app_state.video_capture and app_state.video_capture.isOpened():
            app_state.video_capture.release()
            app_state.video_capture = None
        print("웹캠이 종료되었습니다.")
        
        # 오디오 모니터링 작업 취소
        if app_state.audio_monitoring_task and not app_state.audio_monitoring_task.done():
            app_state.audio_monitoring_task.cancel()
            try:
                await app_state.audio_monitoring_task
            except asyncio.CancelledError:
                print("오디오 모니터링 작업이 취소되었습니다.")
            app_state.audio_monitoring_task = None
        frame_buffer.clear()
        return {"message": "웹캠 및 감지가 중지되었습니다."}
    
    else:
        return JSONResponse(
            status_code=400,
            content={"message": "잘못된 액션입니다. 'START' 또는 'STOP'을 사용하세요."}
        )
        
        
# --- 오디오 모니터링 백그라운드 작업 ---
def detect_cry() -> bool:
    try:
        # 1. 오디오 녹음
        duration = AUDIO_CHUNK_DURATION_SECONDS
        sample_rate = AUDIO_SAMPLE_RATE
        print(f"{duration}초 동안 오디오를 녹음합니다...")

        audio_chunk = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                            channels=1, dtype='int16', device=AUDIO_DEVICE_INDEX)
        sd.wait()

        # 2. 임시 WAV 파일 저장
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        audio_filename = f"audio_{timestamp_str}.wav"
        audio_path = os.path.join(TEMP_AUDIO_FOLDER, audio_filename)

        write_wav(audio_path, sample_rate, audio_chunk)
        print(f"임시 오디오 청크가 저장됨: {audio_path}")

        # 3. 울음소리 예측 (동기 호출)
        is_crying, cry_prob = predict_cry_from_file(audio_path)

        # 4. 결과 처리
        if is_crying:
            print(f"!!! 울음 감지! 확률: {cry_prob:.3f}")
            # 필요한 경우 외부 상태 업데이트 가능
            try:
                os.remove(audio_path)
                print(f"임시 오디오 파일이 삭제되었습니다: {audio_path}")
            except OSError as e:
                print(f"파일 삭제 오류: {e}")

        return is_crying

    except sd.PortAudioError as pae:
        print(f"녹음 중 PortAudioError: {pae}")
        return False
    except Exception as e:
        print(f"오디오 감지 오류: {e}")
        return False    


# 프레임 생성기 수정 (감지 활성화 상태 확인)
async def generate_frames():
    """웹캠에서 프레임을 읽고 처리하여 스트리밍"""

    while True:
        # 감지가 비활성화 상태면 대기
        if not app_state.is_detection_active or app_state.video_capture is None:
            await asyncio.sleep(1)
            continue
            
        # 웹캠이 열려 있지 않으면 다시 시도
        if not app_state.video_capture.isOpened():
            await asyncio.sleep(1)
            continue
            
        success, frame = app_state.video_capture.read()
        if not success:
            print("프레임을 잡는 데 실패했습니다. 다시 시도합니다...")
            await asyncio.sleep(1.0)
            continue

        try:
            # ✅ 프레임을 임시 저장
            await temporary_save(frame, frame_buffer)  # 프레임을 버퍼에 저장

            # 프레임 처리 (YOLO 추론) 및 넘어짐 감지
            processed_frame, fall_detected, person_count = process_frame(frame.copy())

            # 인원 초과 알림
            if person_count > MAX_PERSON_COUNT and app_state.userId:
                current_time = time.time()
                if current_time - app_state.last_overcrowd_time > 30:
                    print()
                    print("인원 초과!!! 알림을 보냅니다 ")
                    print()
                    # 타이머를 먼저 갱신해 중복 방지
                    app_state.last_overcrowd_time = current_time

                    try:
                        await send_overcrowd_notification(person_count, app_state.userId)
                    except Exception as e:
                        print()
                        print("인원 초과 알림 전송 실패: ", str(e))
                        print()
                        
            current_time = time.time() # 일반 현재 시간

            if fall_detected:
                app_state.last_fall_t = datetime.datetime.now().strftime("%M:%S") # Update time of last fall event
                print(f"Fall detected at {app_state.last_fall_t}. ", end="")

                if current_time >= app_state.cooldown_until_next_audio_check:
                    print()
                    print("넘어짐 감지 및 오디오 확인 조건 충족. 오디오 분석 시도.", end=" ")
                    print()
                    # *다음* 시도에 대한 쿨다운을 즉시 설정하세요.
                    app_state.cooldown_until_next_audio_check = current_time + app_state.MIN_AUDIO_CHECK_INTERVAL

                    # detect_cry()는 동기식임. AUDIO_CHUNK_DURATION_SECONDS 동안 차단됨됨.
                    if detect_cry(): 
                        print("넘어짐 및 울음소리 감지됨. 비디오 저장 및 알림 전송 중...", end=" ")
                        # 비동기 작업 중에 deque가 변경될 수 있으므로 저장을 위해 deque의 목록 복사본을 전달.
                        video_path = save_video_from_buffer(list(frame_buffer)) 
                        if video_path: # video_path가 None이 아니거나 비어 있는지 확인
                            await send_video_notification(video_path, app_state.userId)
                        else:
                            print("비디오 저장 실패 또는 경로 없음.")
                    else:
                        print("넘어짐 감지됨, 울음소리 감지되지 않음.")
                    
                    next_check_time_str = datetime.datetime.fromtimestamp(app_state.cooldown_until_next_audio_check).strftime('%H:%M:%S')
                    print(f"오디오 분석 완료. 다음 확인 가능 시간: 약 {next_check_time_str} ( {app_state.MIN_AUDIO_CHECK_INTERVAL}초 후)")
                else:
                    wait_time = app_state.cooldown_until_next_audio_check - current_time
                    print(f"넘어짐 감지됨, 현재 오디오 확인 대기 중 (~{wait_time:.1f}초 남음).", end="")
            
            # 더 나은 로그 가독성을 위해 낙하 관련 메시지가 인쇄된 경우 줄 바꿈을 확인.
            if fall_detected:
                print() 
            # --- END OF LOGIC ---

            # 프레임 스트리밍
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                print("프레임 인코딩 실패")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            print(f"프레임 처리 오류: {e}")
            await asyncio.sleep(1.0)

        #if len(frame_buffer)<=90:
        await asyncio.sleep(0.01)  # 프레임 간 간격



# --- 라우트 정의 ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

    
@app.get("/video_feed")
async def video_feed():
    if not app_state.is_detection_active or app_state.video_capture is None:
        return Response(content="감지가 활성화되지 않았거나 웹캠을 사용할 수 없습니다", status_code=503)
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# --- (선택 사항) 애플리케이션 상태 확인 엔드포인트 ---
@app.get("/status")
async def get_status():
    webcam_open = app_state.video_capture is not None and app_state.video_capture.isOpened()
    try:
        num_temp_files = len([f for f in os.listdir(TEMP_IMAGE_FOLDER) if os.path.isfile(os.path.join(TEMP_IMAGE_FOLDER, f))])
    except FileNotFoundError:
        num_temp_files = 0
    except Exception as e:
        print(f"임시 폴더 상태 확인 오류: {e}")
        num_temp_files = "Error"
    
    audio_task_running = False
    if app_state.audio_monitoring_task:
        audio_task_running = not app_state.audio_monitoring_task.done()

    return {
        "webcam_status": "Open" if webcam_open else "Closed/Error",
        "detection_active": app_state.is_detection_active,
        "userId": app_state.userId,
        "temp_files_count": num_temp_files,
        "audio_monitoring_status": "Running" if audio_task_running else "Stopped",
        "audio_monitoring_active": app_state.audio_monitoring_task is not None and not app_state.audio_monitoring_task.done()
    }

# -- 프레임을 메모리에 JPEG 형식으로 저장하는 함수 (temporary_save) --
async def temporary_save(frame, frame_buffer):
    """프레임을 메모리에 저장 (인코딩 없이)"""
    try:
        if len(frame_buffer) >= MAX_BUFFER_SIZE:
            frame_buffer.popleft()  # 괄호 추가
        frame_buffer.append(frame.copy())

        now_print = datetime.datetime.now().strftime("%M:%S")
        # print(f" temporary_save() 함수 호출됨 : {now_print}, 프레임 저장 완료, 현재 버퍼 크기: {len(frame_buffer)}")
    except Exception as e:
        print(f"프레임 저장 중 오류 발생: {e}")

def save_video_from_buffer(frame_buffer):
    # frame_buffer를 넘겨서 save_video 호출
    video_path = save_video(frame_buffer)
    return video_path

@app.post("/test_fall")
async def test_detect_fall():
    print("test 엔드포인트 호출 됨")
    video_url = save_video_from_buffer(frame_buffer)
    print("test에서 저장 완료")
    return {"video_url": video_url}