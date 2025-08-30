import requests

# 서버 URL 및 엔드포인트 설정
url = "http://127.0.0.1:8000/control_camera"  # uvicorn으로 실행된 서버의 기본 주소와 엔드포인트

# JSON 데이터 정의
data = {
    "cctvAddress": 0,    # "http://웹캠_주소"
    "userId": "qwer123",
    "action": "STOP"
}

# 요청 헤더 설정
headers = {
    "Content-Type": "application/json"
}

# POST 요청 보내기
response = requests.post(url, json=data, headers=headers)

# 응답 출력
print("Status Code:", response.status_code)
print("Response Body:", response.json())
