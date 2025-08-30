# yolo_inference.py (최종 수정 버전)

import httpx
import time
import os
import datetime
import json

from config import (
    MAX_PERSON_COUNT, SPRING_BOOT_OVERCROWD_URL,
    NOTIFICATION_COOLDOWN_SECONDS, SPRING_BOOT_FALL_URL
)

# --- 알림 관련 변수 ---
last_notification_time = 0
last_overcrowd_notification_time = 0

async def send_overcrowd_notification(person_count: int, userId: str):
    """인원 초과 알림 전송 (안전하게 수정)"""
    global last_overcrowd_notification_time
    current_time = time.time()

    if current_time - last_overcrowd_notification_time < NOTIFICATION_COOLDOWN_SECONDS:
        print("인원 초과 알림 쿨다운 중...")
        return False

    # 함수가 호출될 때마다 독립적인 클라이언트를 생성
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            payload = {
                "title": "인원 초과 경고",
                "content": f"현재 인원: {person_count}명 (최대 {MAX_PERSON_COUNT}명)",
                "userId": userId,
                "time": datetime.datetime.now().isoformat(),
            }
            # 새로 만든 'client'를 사용하여 요청
            response = await client.post(SPRING_BOOT_OVERCROWD_URL, json=payload)
            response.raise_for_status()

            print(f"✅ 인원 초과 알림 전송 성공: {person_count}명")
            last_overcrowd_notification_time = current_time
            return True
        except Exception as e:
            print(f"❌ 인원 초과 알림 전송 실패: {e}")
            last_overcrowd_notification_time = current_time
            return False

async def send_video_notification(video_path: str, userId: str):
    """넘어짐, 울음소리 알림 및 비디오 전송 (dto 파트 추가)"""
    global last_notification_time
    current_time = time.time()

    if current_time - last_notification_time < 30: # 쿨다운
        print("넘어짐 및 울음소리 알림 쿨다운 중...")
        return False

    if not os.path.exists(video_path):
        print(f"❌ 영상 파일이 존재하지 않습니다: {video_path}")
        return False

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            with open(video_path, 'rb') as f:
                # 1. DTO에 담을 데이터를 딕셔너리로 생성성
                dto_data = {
                    'title': "울음 + 넘어짐 감지",
                    'content': "아이의 울음과 넘어짐이 동시에 감지되었습니다.",
                    'userId': userId,
                    'time': datetime.datetime.now().isoformat(),
                    'isFell': True,
                }

                # 2. multipart 요청에 포함될 파트들을 구성
                files = {
                    # 파일 파트
                    'videoFile': (os.path.basename(video_path), f, 'video/mp4'),
                    
                    # DTO 파트: 딕셔너리를 JSON 문자열로 변환하여 추가
                    # (파일명은 None, 콘텐트 타입은 'application/json'으로 지정)
                    'dto': (None, json.dumps(dto_data), 'application/json')
                }
                
                # 3. data= 파라미터 없이 files= 파라미터만 사용하여 요청을 보냄냄
                response = await client.post(SPRING_BOOT_FALL_URL, files=files)
                response.raise_for_status()

            print(f"✅ 동영상 알림 전송 완료! Status: {response.status_code}")
            last_notification_time = current_time
            return True

        except FileNotFoundError:
            print(f"❌ 동영상 파일 '{video_path}'을(를) 찾을 수 없습니다.")
            return False
        except httpx.HTTPStatusError as e:
            print(f"❌ 동영상 알림 전송 실패 (HTTP 오류): {e.response.status_code} - {e.response.text}")
            last_notification_time = current_time
            return False
        except Exception as e:
            print(f"❌ 동영상 알림 전송 실패 (일반 오류): {e}")
            last_notification_time = current_time # 실패해도 쿨다운을 적용하여 연속적인 시도를 방지
            return False
