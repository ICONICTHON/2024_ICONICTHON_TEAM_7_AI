import socket
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json
import paho.mqtt.client as mqtt

# MQTT 설정
MQTT_BROKER = 'donggukseoul.com'  # 예: 'mqtt.eclipse.org'
MQTT_PORT = 1883
MQTT_TOPIC = 'sensor/data'  # 원하는 주제 이름
MQTT_CLIENT_ID = 'PythonSensorServer'

# MQTT 클라이언트 설정
mqtt_client = mqtt.Client(MQTT_CLIENT_ID)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# 모델 로드 함수
def load_model(sensor_name):
    model = joblib.load(f'{sensor_name}_model.pkl')
    feature_order = model.feature_names_in_
    return model, feature_order

# 예측 함수
def predict_classroom_rating(date, time, model, feature_order, classroom_number):
    timestamp = pd.to_datetime(f"{date} {time}")
    features = {
        'hour': timestamp.hour,
        'day_of_week': timestamp.dayofweek,
        'month': timestamp.month,
        'year': timestamp.year,
        'day_of_year': timestamp.dayofyear,
        'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
        'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
        'day_of_week_sin': np.sin(2 * np.pi * timestamp.dayofweek / 7),
        'day_of_week_cos': np.cos(2 * np.pi * timestamp.dayofweek / 7),
        'classroom_number': str(classroom_number)
    }
    X = pd.DataFrame([features])
    X = X[feature_order]
    return model.predict(X)[0]

# 모델 로드
model_noise, feature_order_noise = load_model('noise')
model_pm25, feature_order_pm25 = load_model('pm25')
model_temperature, feature_order_temperature = load_model('temperature')
model_tvoc, feature_order_tvoc = load_model('tvoc')
model_humidity, feature_order_humidity = load_model('humidity')

# 서버 설정
HOST = '0.0.0.0'
PORT = 7592

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"서버가 시작되었습니다. 포트 {PORT}에서 연결 대기 중...")

while True:
    client_socket, client_address = server_socket.accept()
    print(f"클라이언트 {client_address}에 연결되었습니다.")
    try:
        while True:
            current_datetime = datetime.now()
            date = current_datetime.strftime('%Y-%m-%d')
            time_str = current_datetime.strftime('%H:%M')
            timestamp_str = current_datetime.strftime('%Y-%m-%d %H:%M')

            classroom_numbers = [1116, 1120, 3101, 3115, 3147, 3173, 4142, 4147, 5145, 5147, 6119, 6144]
            exceeding_classrooms = []
            for classroom_number in classroom_numbers:
                noise = predict_classroom_rating(date, time_str, model_noise, feature_order_noise, classroom_number)
                pm25 = predict_classroom_rating(date, time_str, model_pm25, feature_order_pm25, classroom_number)
                temp = predict_classroom_rating(date, time_str, model_temperature, feature_order_temperature, classroom_number)
                tvoc = predict_classroom_rating(date, time_str, model_tvoc, feature_order_tvoc, classroom_number)
                humidity = predict_classroom_rating(date, time_str, model_humidity, feature_order_humidity, classroom_number)
                
                exceeding_values = {}
                if noise > 3:
                    exceeding_values['소음'] = noise
                if pm25 > 3:
                    exceeding_values['PM2.5'] = pm25
                if temp > 3:
                    exceeding_values['온도'] = temp
                if tvoc > 3:
                    exceeding_values['TVOC'] = tvoc
                if humidity > 3:
                    exceeding_values['습도'] = humidity
                if exceeding_values:
                    exceeding_classrooms.append({
                        '강의실': classroom_number,
                        '이상 수치': exceeding_values
                    })

            sensor_data = json.dumps(exceeding_classrooms, ensure_ascii=False)

            # TCP로 아두이노에 데이터 전송
            client_socket.sendall(sensor_data.encode())
            print(f"센서 데이터 전송 (TCP): {sensor_data}")

            # MQTT 브로커에 데이터 전송
            mqtt_client.publish(MQTT_TOPIC, sensor_data)
            print(f"센서 데이터 전송 (MQTT): {sensor_data}")
            
            time.sleep(60)  

    except (ConnectionResetError, BrokenPipeError):
        print("클라이언트와의 연결이 끊어졌습니다.")
    finally:
        client_socket.close()
