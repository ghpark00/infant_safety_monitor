# Infant Safety Monitor - FastAPI AI Backend

AI 유아 안전 모니터링 및 실시간 대응 시스템의 FastAPI 기반 분석 서버입니다.
RTSP/웹캠 영상을 받아 낙상과 인원 초과를 분석하고, 낙상 감지 시 현장 오디오에서 울음 여부를 함께 판단해 Spring Boot 서버로 알림과 영상 클립을 전송합니다.

## 주요 기능

- 실시간 영상 수신: RTSP 주소 또는 웹캠 입력 처리
- 실시간 AI 분석: YOLO 기반 사람 탐지, SPPE 관절점 추출, ST-GCN 낙상 감지
- 인원 초과 감지: 설정된 최대 인원 초과 시 알림 전송
- 울음 감지: 낙상 이벤트 발생 시 YAMNet 임베딩과 이진 분류기로 울음 여부 판단
- 이벤트 영상 저장: 이벤트 전후 프레임 버퍼를 `.mp4` 클립으로 저장
- 원격 제어 API: `/control_camera`로 분석 시작/중지
- 영상 스트리밍: `/video_feed`에서 바운딩 박스와 분석 결과가 반영된 MJPEG 스트림 제공

## 폴더 구조

```text
.
├── ai_backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app, route, streaming loop
│   │   └── templates/           # 간단한 스트리밍 확인 페이지
│   ├── inference/
│   │   ├── fall_detection_pipeline.py
│   │   └── audio_inference.py
│   ├── services/
│   │   └── notifications.py     # Spring Boot 알림 연동
│   ├── storage/
│   │   └── video.py             # 이벤트 클립 저장
│   ├── scripts/
│   │   ├── process_video.py     # 오프라인 MP4 분석 CLI
│   │   ├── detected_main.py     # 원본 fall-detection 데모 실행 스크립트
│   │   ├── startJSON.py         # /control_camera START 요청 예제
│   │   └── stopJSON.py          # /control_camera STOP 요청 예제
│   ├── model_runtime/           # YOLO, SPPE, ST-GCN 런타임 로더와 시각화 유틸
│   ├── legacy/                  # 예전 GUI/학습 시각화 코드 보관
│   ├── utils/
│   │   └── video_buffer.py
│   └── config.py                # 경로, 임계값, 외부 URL 설정
├── Actionsrecognition/          # ST-GCN action recognition model code
├── Detection/                   # Tiny-YOLO detection model code
├── SPPE/                        # AlphaPose/SPPE pose estimation code
├── Track/                       # tracking utilities
├── models/                      # ignored: model weights and cfg files
├── docs/                        # 프로젝트 산출 문서
├── main.py                      # compatibility entrypoint
└── process_video.py             # compatibility CLI wrapper
```

루트에는 실행 진입점인 `main.py`, `process_video.py`만 남겼습니다. 기존 fall-detection 계열 모델 로더는 `ai_backend/model_runtime/`으로 이동했습니다.

## 실행

```bash
uvicorn ai_backend.api.main:app --reload
```

기존 방식도 동작하도록 `main.py` wrapper를 남겨두었습니다.

```bash
python main.py
```

오프라인 MP4 분석:

```bash
python process_video.py --input test_videos/sample.mp4 --output result_videos/output.mp4
```

알림까지 테스트하려면:

```bash
python process_video.py --input test_videos/sample.mp4 --output result_videos/output.mp4 --notify --userId USER_ID
```

## API

- `POST /control_camera`
  - `action`: `START` 또는 `STOP`
  - `cctvAddress`: 웹캠 인덱스 또는 RTSP 주소
  - `userId`: 알림 대상 사용자 ID
- `GET /video_feed`: 분석 결과가 반영된 MJPEG 스트리밍
- `GET /status`: 카메라와 분석 상태 확인
- `POST /test_fall`: 현재 프레임 버퍼로 테스트 영상 저장

## 환경 변수

`.env`로 Spring Boot 서버 주소를 덮어쓸 수 있습니다.

```env
SPRING_BOOT_FALL_URL=http://localhost:8080/api/notify/fall_cry_file
SPRING_BOOT_OVERCROWD_URL=http://localhost:8080/api/alarms/overcrowd
SPRING_BOOT_CRY_URL=http://localhost:8080/api/notify/cry
```

## 모델 파일

모델 가중치는 `.gitignore`에 의해 Git에 포함하지 않습니다. 기본 경로는 다음과 같습니다.

```text
models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg
models/yolo-tiny-onecls/best-model.pth
models/sppe/fast_res50_256x192.pth
models/TSSTG/tsstg-model.pth
models/my_model.h5
```
