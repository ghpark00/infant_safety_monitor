# audio_inference.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from scipy.io.wavfile import write
from config import (
    CRY_MODEL_PATH, YAMNET_MODEL_HANDLE, AUDIO_SAMPLE_RATE,
    CRY_CONF_THRESHOLD, TEMP_AUDIO_FOLDER
)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# --- 모델 로드 (애플리케이션 시작 시 로드되도록 main.py에서 호출) ---
yamnet_model = None
cry_detection_model = None
last_cry_notification_time = 0

def load_audio_models():
    """오디오 관련 TensorFlow 모델 로드"""
    global yamnet_model, cry_detection_model
    try:
        print("YAMNET 모델을 로딩 중입니다....")
        yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
        print("YAMNet 모델이 로드되었습니다!")

        print(f"{CRY_MODEL_PATH}에서 울음 감지 모델 로드 중")
        # 'compile=False'는 로드 속도를 높일 수 있지만, 모델 구조에 따라 필요할 수 있음
        cry_detection_model = tf.keras.models.load_model(CRY_MODEL_PATH, compile=False)
        print("울음 감지 모델이 로드되었습니다!")
        # 모델 워밍업 (선택 사항, 첫 예측 지연 감소)
        # try:
        #    dummy_embeddings = tf.zeros((1, 1024), dtype=tf.float32) # YAMNet 출력 형태 가정
        #    cry_detection_model.predict(dummy_embeddings)
        #    print("Cry model warmed up.")
        # except Exception as warmup_e:
        #    print(f"Could not warm up cry model: {warmup_e}")

    except Exception as e:
        print(f"오디오 모델 로딩 중 오류 발생: {e}")
        yamnet_model = None
        cry_detection_model = None

# --- 오디오 전처리 함수 (사용자 제공 코드 기반) ---
# @tf.function # 성능 향상을 위해 tf.function 사용 고려
def preprocess_audio(file_path):
    """WAV 파일을 읽고 전처리하여 waveform 반환"""
    try:
        file_data = tf.io.read_file(file_path)
        # decode_wav는 target_sample_rate 인자를 직접 지원하지 않으므로, 로드 후 리샘플링 필요 시 추가
        audio, sample_rate = tf.audio.decode_wav(file_data, desired_channels=1)

        # 샘플 속도 확인 및 변환 (필요 시) - YAMNet은 16kHz 필요
        if sample_rate != AUDIO_SAMPLE_RATE:
            print(f"경고: 입력 샘플링 레이트({sample_rate}Hz)가 목표 샘플링 레이트({AUDIO_SAMPLE_RATE}Hz)와 다릅니다. 리샘플링이 필요하지만 이 기본 버전에는 구현되어 있지 않습니다.")
            # 실제 리샘플링 구현: tfio.audio.resample 또는 librosa 사용 필요
            # 여기서는 단순화를 위해 경고만 출력

        audio = tf.squeeze(audio, axis=-1) # (N, 1) -> (N,)
        # YAMNet 입력 형식에 맞추기 (0과 1 사이 값으로 정규화 - 필요 시)
        # audio = audio / tf.int16.max # WAV가 16비트인 경우
        return audio
    except Exception as e:
        print(f"오디오 파일 {file_path} 전처리 오류: {e}")
        return None

# --- 울음소리 예측 함수 (사용자 제공 코드 기반, 반환값 수정) ---
def predict_cry_from_file(file_path):
    """오디오 파일 경로를 받아 울음소리 여부와 확률 반환"""
    if yamnet_model is None or cry_detection_model is None:
        print("오디오 모델이 로드되지 않아 예측이 불가능합니다.")
        return False, 0.0

    try:
        waveform = preprocess_audio(file_path)
        if waveform is None:
            return False, 0.0
        
        # waveform이 너무 짧으면 YAMNet에서 오류 발생 가능성 있음
        min_length = int(0.96 * AUDIO_SAMPLE_RATE) # YAMNet 기본 윈도우 크기 관련
        if tf.shape(waveform)[0] < min_length:
            print(f"경고: 오디오 파형이 너무 짧습니다({tf.shape(waveform)[0]}개 샘플). 예측을 건너뜁니다.")
            # 패딩 또는 다른 처리 필요 가능성
            return False, 0.0


        # YAMNet 임베딩 추출
        # waveform 텐서를 직접 입력으로 사용
        _, embeddings, _ = yamnet_model(waveform) # (N, 1024) 형태의 임베딩 예상

        # 임베딩이 비어 있는지 확인 (짧은 오디오 등)
        if tf.shape(embeddings)[0] == 0:
            print("경고: 오디오에서 내장이 생성되지 않았기 때문에 예측이 건너뛰어졌습니다.")
            return False, 0.0

        # 커스텀 모델 예측 (TF 함수는 기본적으로 배치 입력 가정)
        prediction = cry_detection_model.predict(embeddings, verbose=0) # verbose=0으로 로그 출력 줄임

        # 평균 확률 계산
        cry_probability = np.mean(prediction)

        # 임계값 비교하여 최종 판별
        is_crying = cry_probability > CRY_CONF_THRESHOLD

        # 결과 출력 (선택 사항)
        # if is_crying:
        #     print(f"울음 감지! (평균 확률: {cry_probability:.3f}) in {file_path}")
        # else:
        #     print(f"울음 없음 (평균 확률: {cry_probability:.3f}) in {file_path}")

        return is_crying, cry_probability

    except tf.errors.InvalidArgumentError as tf_err:
        print(f"{file_path}에 대한 예측 중 TensorFlow 인수 오류가 발생했습니다: {tf_err}")
        # 흔한 원인: 오디오 길이, 데이터 타입 문제
        return False, 0.0
    except Exception as e:
        print(f"파일 {file_path}에 대한 Cry 예측 오류: {e}")
        return False, 0.0

# --- 임시 오디오 폴더 확인/생성 함수 ---
def ensure_temp_audio_folder():
    if not os.path.exists(TEMP_AUDIO_FOLDER):
        try:
            os.makedirs(TEMP_AUDIO_FOLDER)
            print(f"임시 오디오 폴더가 생성되었습니다: {TEMP_AUDIO_FOLDER}")
        except OSError as e:
            print(f"임시 폴더 {TEMP_AUDIO_FOLDER}를 생성하는 동안 오류가 발생했습니다: {e}")
            return False
    return True