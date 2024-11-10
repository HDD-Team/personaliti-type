from fastapi import FastAPI, HTTPException, Depends, Security, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import shutil
import bcrypt
import asyncpg
import uvicorn
import pytz
import os
import re
from itertools import zip_longest
from typing import Optional, List
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import magic
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import cv2
import torch.multiprocessing as mp
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score

# uvicorn api.main:app --log-config log_config.yaml --host 0.0.0.0 --port 8001 --reload

mp.set_start_method('spawn', force=True)


# Устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

# Загрузка модели YOLOv8 для позинг-детекции
yolo_model = YOLO("yolov8n-pose.pt").to(device)

# Функция для извлечения ключевых точек позы с использованием YOLOv8
def extract_keypoints(frame):
    results = yolo_model(frame)
    keypoints = []

    # Обработка результатов
    for detection in results[0].keypoints.xy:  # Используем атрибут xy для доступа к координатам [x, y]
        for point in detection:
            keypoints.extend([point[0].item(), point[1].item()])  # Извлекаем x, y координаты

    # Гарантируем, что длина keypoints всегда равна 17 * 2
    if len(keypoints) != 17 * 2:
        keypoints = keypoints[:17 * 2] + [0] * (17 * 2 - len(keypoints))

    return keypoints
#
# def acc(metr):
    # answer = []
    # for output, labels in metr:
        # output = output.tolist()
        # labels = labels.tolist()
        # ans = []
        # for i in len(labels):
            # ans.append(labels[i]-output[i])
        # answer.append(ans)
    # for ans in

class OceanLSTM(nn.Module):
        def __init__(self, input_size=17 * 2, hidden_size=64, output_size=6):  # 2 координаты на ключевую точку
            super(OceanLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)  # 5 выходов для черт OCEAN

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            output = self.fc(h_n[-1])
            return output


class OceanDataset(Dataset):
    def __init__(self, video_paths, sequence_length=10):
        self.video_paths = video_paths
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_paths[idx])
        sequence = []

        while cap.isOpened() and len(sequence) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (640, 640))

                # Преобразование формата с HWC на CHW
            frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
            keypoints = extract_keypoints(frame_tensor)
            sequence.append(keypoints)

        cap.release()

        # Если длина последовательности меньше требуемой, добавляем пустые кадры
        if len(sequence) < self.sequence_length:
            sequence.extend([[0] * 17 * 2] * (self.sequence_length - len(sequence)))

        # Преобразуем в тензоры
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        

        return sequence_tensor

def inference(video_paths):
    model = OceanLSTM(input_size=17 * 2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    PATH = 'api/MODELS_TO_SUBMIT/ocean_model_with_optimizer.pth'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    model.eval()
    dataset = OceanDataset(video_paths)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    all_preds = []
    with torch.no_grad():
            for sequences in dataloader:
                sequences = sequences.to(device)
                output = model(sequences)
                all_preds.extend(output.cpu().tolist().pop(4))
    print('MODEL_PRED --- ', all_preds)
    return all_preds

def ocean_to_mbti_weighted(json_input):
    """
    Функция для преобразования OCEAN черт личности в тип MBTI, используя веса для каждой из четырех характеристик MBTI.
    Также возвращает подходящую профессию на основе определенного типа MBTI.

    Параметры:
    - json_input (str): JSON-строка, содержащая значения для каждого из пяти OCEAN признаков:
        - Openness (Открытость)
        - Conscientiousness (Добросовестность)
        - Extraversion (Экстраверсия)
        - Agreeableness (Дружелюбие)
        - Neuroticism (Нейротизм)

    Возвращаемое значение:
    - JSON-строка, содержащая определенный тип MBTI и подходящую профессию в формате:
        - MBTI_Type: Строка с 4 буквами, представляющая тип личности MBTI
        - Profession: Подходящая профессия для данного типа MBTI
    """

    ocean_scores = json_input
    weights = {
        "E/I": {"Extraversion": 0.6, "Openness": 0.2, "Agreeableness": 0.1, "Conscientiousness": 0, "Neuroticism": 0.1},
        "N/S": {"Extraversion": 0.1, "Openness": 0.7, "Agreeableness": 0, "Conscientiousness": 0.2, "Neuroticism": 0},
        "T/F": {"Extraversion": 0, "Openness": 0.1, "Agreeableness": 0.6, "Conscientiousness": 0.3, "Neuroticism": 0},
        "J/P": {"Extraversion": 0, "Openness": 0.2, "Agreeableness": 0.1, "Conscientiousness": 0.6, "Neuroticism": 0.1}
    }

    mbti_scores = {
        "E": sum(ocean_scores[trait] * weight for trait, weight in weights["E/I"].items()),
        "I": sum((1 - ocean_scores[trait]) * weight for trait, weight in weights["E/I"].items()),
        "N": sum(ocean_scores[trait] * weight for trait, weight in weights["N/S"].items()),
        "S": sum((1 - ocean_scores[trait]) * weight for trait, weight in weights["N/S"].items()),
        "T": sum((1 - ocean_scores[trait]) * weight for trait, weight in weights["T/F"].items()),
        "F": sum(ocean_scores[trait] * weight for trait, weight in weights["T/F"].items()),
        "J": sum(ocean_scores[trait] * weight for trait, weight in weights["J/P"].items()),
        "P": sum((1 - ocean_scores[trait]) * weight for trait, weight in weights["J/P"].items())
    }

    mbti_type = ""
    mbti_type += "E" if mbti_scores["E"] > mbti_scores["I"] else "I"
    mbti_type += "N" if mbti_scores["N"] > mbti_scores["S"] else "S"
    mbti_type += "T" if mbti_scores["T"] > mbti_scores["F"] else "F"
    mbti_type += "J" if mbti_scores["J"] > mbti_scores["P"] else "P"

    mbti_professions = {
        "INTJ": "Научный исследователь",
        "INTP": "Исследователь",
        "ENTJ": "Руководитель проектов",
        "ENTP": "Предприниматель",
        "INFJ": "Психолог",
        "INFP": "Художник",
        "ENFJ": "Учитель",
        "ENFP": "Актёр",
        "ISTJ": "Бухгалтер",
        "ISFJ": "Медсестра",
        "ESTJ": "Менеджер",
        "ESFJ": "Учитель",
        "ISTP": "Инженер",
        "ISFP": "Дизайнер",
        "ESTP": "Продавец",
        "ESFP": "Актёр"
    }
    profession = mbti_professions.get(mbti_type, "Профессия не найдена")
    output = {
        "MBTI_Type": mbti_type,
        "Profession": profession
    }
    print('OCEAN RESULT --- ', output)
    return json.dumps(output)

app = FastAPI()

# Инициализация библиотеки magic
mime = magic.Magic(mime=True)

# Получаем параметры базы данных
load_dotenv('docker_env.env')

db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все домены
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Разрешаемые методы (GET, POST)
    allow_headers=["*"],  # Разрешить все заголовки
)

# Настройки для JWT
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Настройки для хэширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Настройки для OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Функция для хеширования пароля
async def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Функция для проверки пароля
async def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# Функция для создания токена доступа
async def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Подключение к базе данных
DATABASE_URL = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}?options=-c timezone=Etc/GMT-3"

# Глобальный пул подключений
pool = None

@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL)

    # Загрузка модели YOLOv8
    yolo_model = YOLO("yolov8n-pose.pt").to(device)

@app.on_event("shutdown")
async def shutdown():
    await pool.close()


class VideoPaths(BaseModel):
    video_paths: List[str]

# Модель данных для регистрации
class UserRegistration(BaseModel):
    first_name: str
    last_name: str
    phone: str
    login: str
    password: str
    role: str

# Модель для авторизации
class UserLogin(BaseModel):
    login: str
    password: str

class UserInfo(BaseModel):
    user_id: int

class EmployerInfo(BaseModel):
    user_id: int

# Кастомная форма для авторизации
class CustomOAuth2PasswordRequestForm(OAuth2PasswordRequestForm):
    def __init__(self, login: str, password: str):
        super().__init__(username=login, password=password)


@app.post('/api/start-model')
async def start_model(video_data: VideoPaths):
    try:
        video_paths = video_data.video_paths
        # Проверяем, что видеофайлы существуют
        for path in video_paths:
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail=f"Видео файл {path} не найден")

        # Вызываем функцию inference из файла модели
        model_results = inference(video_paths)

        # Обрабатываем каждый результат через функцию ocean_to_mbti_weighted
        processed_results = []
        traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        for result in model_results:
            result_dict = dict(zip(traits, result))
            # Нормализуем значения в диапазон [0, 1], если это необходимо
            # Здесь предполагается, что результаты модели уже в нужном диапазоне
            processed_result = ocean_to_mbti_weighted(result_dict)
            processed_results.append({
                "ocean_scores": result_dict,
                "mbti_result": processed_result
            })

        # Возвращаем результаты клиенту
        return JSONResponse(content={"results": processed_results})
    except Exception as e:
        print('START MODEL ERROR --- ', e)
        raise HTTPException(status_code=500, detail=str(e))


# Эндпоинт для регистрации
@app.post("/api/register")
async def register(user: UserRegistration):
    print(user)
    # Проверка валидности роли
    if user.role not in ['candidate', 'employer']:
        raise HTTPException(status_code=400, detail="Invalid role")

    async with pool.acquire() as conn:
        try:
            # Хеширование пароля
            hashed_password = await get_password_hash(user.password)

            # Вставка пользователя
            user_id = await conn.fetchval("""
                INSERT INTO users (login, password, role)
                VALUES ($1, $2, $3) RETURNING user_id
            """, user.login, hashed_password, user.role)

            # Вставка данных кандидата или работодателя
            if user.role == 'candidate':
                await conn.execute("""
                    INSERT INTO candidates (candidate_id, first_name, last_name, phone)
                    VALUES ($1, $2, $3, $4)
                """, user_id, user.first_name, user.last_name, user.phone)
            elif user.role == 'employer':
                await conn.execute("""
                    INSERT INTO employers (employer_id, first_name, last_name, phone)
                    VALUES ($1, $2, $3, $4)
                """, user_id, user.first_name, user.last_name, user.phone)

            return {"success": True}
        except asyncpg.UniqueViolationError as e:
            print(e)
            raise HTTPException(status_code=500, detail={"success": False, "message": "Email already exists."})
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail={"success": False, "message": str(e)})

# Эндпоинт для авторизации
@app.post("/api/token")
async def login(form_data: CustomOAuth2PasswordRequestForm = Depends()):
    print(form_data)
    async with pool.acquire() as conn:
        try:
            # Получение пользователя из базы данных
            user_data = await conn.fetchrow("""
                SELECT user_id, password FROM users WHERE login = $1
            """, form_data.username)

            if not user_data:
                return HTTPException(status_code=400, detail={"success": False, "message": "Incorrect login or password"})

            check_pass = await verify_password(form_data.password, user_data['password'])
            # Проверка пароля
            if not check_pass:
                return HTTPException(status_code=400, detail={"success": False, "message": "Incorrect login or password"})

            user_role = await conn.fetchrow("""
                SELECT role FROM users WHERE login = $1
            """, form_data.username)

            return {"success": True, "role": user_role["role"], "id": user_data[0]}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail={"success": False, "message": str(e)})

@app.post('/api/card')
async def card(user_id: UserInfo):
    print('/card User ID - ', user_id)
    async with pool.acquire() as conn:
        try:
            # Получение данные пользователя из базы данных
            user_data = await conn.fetchrow("""
                SELECT login, first_name, last_name, phone, profile_video_url, personality_type, profile_photo_url FROM candidates WHERE candidate_id = $1
            """, user_id.user_id)
            if user_data:
                return JSONResponse(status_code=200, content={
                    "login": user_data[0],
                    "first_name": user_data[1],
                    "last_name": user_data[2],
                    "phone": user_data[3],
                    "profile_video_url": user_data[4],
                    "personality_type": user_data[5],
                    "profile_photo_url": user_data[6],
                    "suitableVacancies": []
                })
            else:
                return HTTPException(status_code=404, detail={"message": "User data not found"})
        except Exception as e:
            raise HTTPException(status_code=500, detail={"message": str(e)})
        

@app.post('/api/employer-card')
async def emloyer_card(user_id: UserInfo):
    print('/employer-card User ID - ', user_id.user_id)
    async with pool.acquire() as conn:
        try:
            # Получаем данные из таблицы employers
            employer_data = await conn.fetchrow("""
                SELECT
                    login,
                    first_name,
                    last_name,
                    phone
                FROM
                    employers
                WHERE
                    employer_id = $1
            """, user_id.user_id)

            # Получаем данные из таблицы employers_videos
            videos_data = await conn.fetch("""
                SELECT
                    upload_id,
                    videos_url,
                    upload_date,
                    upload_name
                FROM
                    employers_videos
                WHERE
                    employer_id = $1
            """, user_id.user_id)
            if employer_data:

                # Преобразуем данные в нужный формат
                response_data = {
                    "login": employer_data["login"],
                    "first_name": employer_data["first_name"],
                    "last_name": employer_data["last_name"],
                    "phone": employer_data["phone"],
                    "uploads": {
                        record["upload_id"]: {
                            "upload_id": record["upload_id"],
                            "videos_url": record["videos_url"],
                            "upload_date": str(record["upload_date"]),
                            "upload_name": record["upload_name"]
                        }
                        for record in videos_data
                    }
                }

                # Возвращаем JSONResponse
                return JSONResponse(status_code=200, content=response_data)
            else:
                raise HTTPException(status_code=404, detail={"message": "User data not found"})
        except Exception as e:
            if not isinstance(e, HTTPException):
                print(e)
                raise HTTPException(status_code=500, detail={"message": str(e)})

def get_video_range(start: int, end: int, file_path: str):
    with open(file_path, "rb") as video_file:
        video_file.seek(start)
        while start < end:
            chunk_size = min(1024 * 1024, end - start)
            data = video_file.read(chunk_size)
            if not data:
                break
            yield data
            start += chunk_size

@app.post("/api/card-video")
async def video_endpoint(user_id: UserInfo):
    try:
        async with pool.acquire() as conn:
            user_video_path = await conn.fetchrow("""
                SELECT profile_video_url FROM candidates WHERE candidate_id = $1
            """, user_id.user_id)
            if not user_video_path or not user_video_path['profile_video_url']:
                return HTTPException(status_code=404, detail="Video not found")

            video_url = f"/stream-video/{user_id.user_id}"
            return {"videoUrl": video_url}
    except Exception as e:
        print('ERROR in /card-video - ', e)
        raise HTTPException(status_code=500, detail={"message": str(e)})

@app.get("/api/stream-video/{user_id}")
async def stream_video(user_id: int, request: Request):
    try:
        async with pool.acquire() as conn:
            user_video_path = await conn.fetchrow("""
                SELECT profile_video_url FROM candidates WHERE candidate_id = $1
            """, user_id)
            if not user_video_path or not user_video_path['profile_video_url']:
                return HTTPException(status_code=404, detail="Video not found")

            file_size = os.path.getsize(user_video_path['profile_video_url'])
            range_header = request.headers.get("range")

            if range_header is None:
                return StreamingResponse(get_video_range(0, file_size, user_video_path['profile_video_url']), media_type="video/mp4")

            range_value = range_header.strip().split("=")[-1]
            start_str, end_str = range_value.split("-")
            start = int(float(start_str)) if start_str else 0
            end = int(float(end_str)) if end_str else file_size - 1
            end = min(end, file_size - 1)

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(end - start + 1),
                "Content-Type": "video/mp4",
            }

            return StreamingResponse(get_video_range(start, end + 1, user_video_path['profile_video_url']), headers=headers, status_code=206)
    except Exception as e:
        print('ERROR in /stream-video - ', e)
        raise HTTPException(status_code=500, detail={"message": str(e)})


def employer_get_video_range(start: int, end: int, file_paths: list):
    for file_path in file_paths:
        with open(file_path, "rb") as video_file:
            video_file.seek(start)
            while start < end:
                chunk_size = min(1024 * 1024, end - start)
                data = video_file.read(chunk_size)
                if not data:
                    break
                yield data
                start += chunk_size



@app.post("/api/employer-card-video")
async def video_endpoint(user_id: UserInfo, upload_id: int):
    try:
        async with pool.acquire() as conn:
            user_video_paths = await conn.fetchrow("""
                SELECT videos_url, video_descriptions FROM employers_videos WHERE employer_id = $1 AND upload_id = $2
            """, user_id.user_id, upload_id)
            print('[/employer-card-video] All videos paths - ', user_video_paths['videos_url'])
            print('[/employer-card-video] All videos desriptions - ', user_video_paths['video_descriptions'])

            if not user_video_paths or not user_video_paths['videos_url'][0]:
                raise HTTPException(status_code=404, detail="Video not found")
            
            video_urls = [f"/employer-stream-video/{user_id.user_id}/{upload_id}/{index}" for index, _ in enumerate(user_video_paths['videos_url'])]
            video_names = [f"{name.split('/')[-1]}" for name in user_video_paths['videos_url']]
            videos_description = [description for description in user_video_paths['video_descriptions'] or [None] * len(video_urls)]
            video_urls = [
                {"videoUrl": url, "videName": name, "videDescription": description}
                for url, name, description in zip_longest(video_urls, video_names, videos_description, fillvalue='')
              ]
            return {"videoUrls": video_urls}
    except Exception as e:
        if not isinstance(e, HTTPException):
            print('ERROR in /employer-card-video - ', e)
            raise HTTPException(status_code=500, detail={"message": str(e)})

@app.get("/api/employer-stream-video/{user_id}/{upload_id}/{index}")
async def stream_video(user_id: int, upload_id: int, index: int, request: Request):
    try:
        async with pool.acquire() as conn:
            user_video_paths = await conn.fetchrow("""
                SELECT videos_url FROM employers_videos WHERE employer_id = $1 AND upload_id = $2
            """, user_id, upload_id)
            if not user_video_paths or index >= len(user_video_paths['videos_url']):
                raise HTTPException(status_code=404, detail="Video not found")

            file_path = user_video_paths['videos_url'][index]
            file_size = os.path.getsize(file_path)
            range_header = request.headers.get("range")

            if range_header is None:
                return StreamingResponse(employer_get_video_range(0, file_size, [file_path]), media_type="video/mp4")

            range_value = range_header.strip().split("=")[-1]
            start_str, end_str = range_value.split("-")
            start = int(float(start_str)) if start_str else 0
            end = int(float(end_str)) if end_str else file_size - 1
            end = min(end, file_size - 1)

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(end - start + 1),
                "Content-Type": "video/mp4",
            }

            return StreamingResponse(employer_get_video_range(start, end + 1, [file_path]), headers=headers, status_code=206)
    except Exception as e:
        if not isinstance(e, HTTPException):
            print('ERROR in /employer-stream-video - ', e)
            raise HTTPException(status_code=500, detail={"message": str(e)})


@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...), user_id: str = Form(...)):
    print('[/upload-video] User ID - ', user_id)
    try:
        async with pool.acquire() as conn:
            if user_id:
                # Получение данные пользователя из базы данных
                user_check = await conn.fetchrow("""
                    SELECT login FROM candidates WHERE candidate_id = $1
                """, int(user_id))
                if user_check:
                    print('[/upload-video] - True User')
                    # Проверка MIME-типа файла
                    file_mime_type = mime.from_buffer(file.file.read(1024))
                    file.file.seek(0)  # Сброс позиции чтения файла

                    if not file_mime_type.startswith('video/'):
                        raise HTTPException(status_code=400, detail={"success": False, "message": "Uploaded file is not a video"})
                    else:
                        # Сохранение файла на сервере
                        with open(f"uploads_files/{file.filename}", "wb") as buffer:
                            shutil.copyfileobj(file.file, buffer)
                            print(f'[/upload-video] - Video {file.filename} saved')

                        video_path = f"uploads_files/{file.filename}"
                        await conn.fetchval("""
                            UPDATE candidates
                            SET profile_video_url = $1
                            WHERE candidate_id = $2
                        """, video_path, int(user_id))
                        print('[/upload-video] - Database updated')
                        return JSONResponse(status_code=200, content={"success": True})
                else:
                    print('[/upload-video] - False User')
                    return HTTPException(status_code=500, detail={"success": False, "message": "User not found"})
            else:
                print('[/upload-video] - False metadata')
                return HTTPException(status_code=500, detail={"success": False, "message": "Uploaded file dont have metadata"})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"success": False, "message": str(e)})
    

@app.post("/api/employer-upload-video")
async def upload_video(upload_id: Optional[int] = Form(default=None), upload_date: Optional[datetime] = Form(default=None), file: UploadFile = File(...), user_id: str = Form(...)):
    print('[/employer-upload-video] User ID - ', user_id)
    upload_id_update = upload_id
    try:
        
        async with pool.acquire() as conn:
            if user_id and upload_date:
                date_format = "%Y-%m-%d %H:%M:%S.%f%z"
                date_format_lite = "%Y-%m-%d %H:%M:%S"
                upload_name_default = datetime.strftime(upload_date, date_format_lite)
                date_name_part = re.sub(r'[^a-zA-Z0-9_-]', '_', str(upload_date))
                folder_name = f'uploads_files/employer_{user_id}_{date_name_part}'
                print('[/employer-upload-video] Upload date - ', upload_date)
                # Получение данные пользователя из базы данных
                user_check = await conn.fetchrow("""
                    SELECT login FROM employers WHERE employer_id = $1
                """, int(user_id))
                if user_check:
                    print('[/employer-upload-video] - True User')

                    # Проверка MIME-типа файла
                    file_mime_type = mime.from_buffer(file.file.read(1024))
                    file.file.seek(0)  # Сброс позиции чтения файла

                    if not file_mime_type.startswith('video/'):
                        raise HTTPException(status_code=400, detail={"success": False, "message": "Uploaded file is not a video"})
                    else:
                        file_name = f'employer_{file.filename}'
                        video_path = f"{folder_name}/{file_name}"

                        
                        
                        check_upload_date = await conn.fetch("""
                            SELECT upload_date
                            FROM employers_videos
                            WHERE employer_id = $1
                        """, int(user_id))
                        print(f'[/employer-upload-video] All upload dates: {check_upload_date}')
                        if check_upload_date:
                            # check_upload_date = [datetime.strptime(date_str, date_format).date() for date_str in check_upload_date[0]['upload_date']]
                            # print(check_upload_date)
                            moscow_tz = pytz.timezone('Europe/Moscow')
                            check_upload_date_list = [record['upload_date'].astimezone(moscow_tz) for record in check_upload_date]
                            print('1')
                            if upload_date in check_upload_date_list:
                                videos_path_list = await conn.fetchrow("""
                                    SELECT videos_url
                                    FROM employers_videos
                                    WHERE upload_date = $1
                                """, upload_date)
                                print(type(videos_path_list['videos_url']))
                                if videos_path_list:
                                    videos_path_list['videos_url'].append(video_path)
                                    videos_path_list_update = videos_path_list['videos_url']

                                    if not os.path.exists(video_path):
                                        with open(video_path, "wb") as buffer:
                                            shutil.copyfileobj(file.file, buffer)
                                            
                                            print(f'[/employer-upload-video] - Video {file_name} saved')
                                    else:
                                        print(f'[/employer-upload-video] - Video {file_name} already exists')
                                        raise Exception(f'Video {file_name} already exists')

                                    await conn.execute("""
                                        UPDATE employers_videos
                                        SET videos_url = $1
                                        WHERE upload_date = $2
                                    """, videos_path_list_update, upload_date)
                                    print('[/employer-upload-video] Updated videos urls list. List now: ', videos_path_list)
                                else:
                                    print(f'[/employer-upload-video] ERROR Not found videos urls list from User ID: {user_id}')
                                    raise Exception("Not found videos urls list from user")
                            else:
                                
                                # Сохранение файла на сервере
                                if not os.path.exists(folder_name):
                                    os.makedirs(folder_name)
                                    if not os.path.exists(video_path):
                                        with open(video_path, "wb") as buffer:
                                            shutil.copyfileobj(file.file, buffer)
                                            
                                            print(f'[/employer-upload-video] - Video {file_name} saved')
                                    else:
                                        print(f'[/employer-upload-video] - Video {file_name} already exists')
                                        raise Exception(f'Video {file_name} already exists')
                                else:
                                    print(f'[/employer-upload-video] - Folder {folder_name} already exists')
                                    raise Exception(f'Folder {folder_name} already exists')
                                
                                videos_upload = await conn.fetchrow("""
                                    SELECT upload_id
                                    FROM employers_videos
                                    WHERE employer_id = $1
                                    ORDER BY upload_id DESC
                                    LIMIT 1
                                """, int(user_id))

                                upload_id = videos_upload['upload_id'] + 1

                                await conn.execute("""
                                    INSERT INTO employers_videos (employer_id, upload_id, videos_url, upload_date, upload_name)
                                    VALUES ($1, $2, $3, $4, $5)
                                """, int(user_id), upload_id, [video_path], upload_date, upload_name_default)
                                print(f'[/employer-upload-video] Add Upload ID - {upload_id}. List now: {[video_path]}')
                        else:
                            
                            # print('[/employer-upload-video] Last upload ID - ', videos_upload['upload_id'])
                            # if not videos_upload['upload_id']:
                            # Сохранение файла на сервере
                            if not os.path.exists(folder_name):
                                os.makedirs(folder_name)
                                if not os.path.exists(video_path):
                                    with open(video_path, "wb") as buffer:
                                        shutil.copyfileobj(file.file, buffer)
                                        
                                        print(f'[/employer-upload-video] - Video {file_name} saved')
                                else:
                                    print(f'[/employer-upload-video] - Video {file_name} already exists')
                                    raise Exception(f'Video {file_name} already exists')
                            else:
                                print(f'[/employer-upload-video] - Folder {folder_name} already exists')
                                raise Exception(f'Folder {folder_name} already exists')
                            
                            upload_id = 1
                            await conn.execute("""
                                INSERT INTO employers_videos (employer_id, upload_id, videos_url, upload_date, upload_name)
                                VALUES ($1, $2, $3, $4, $5)
                            """, int(user_id), upload_id, [video_path], upload_date, upload_name_default)
                            print(f'[/employer-upload-video] Add first upload. List now: {[video_path]}')

                        print('[/employer-upload-video] - Database updated')
                        return JSONResponse(status_code=200, content={"success": True})
                else:
                    print('[/employer-upload-video] - False User')
                    raise Exception("User not found")
            elif upload_id:
                print('[/employer-upload-video] Add video an existing card ID: ', upload_id)
                # Получение данные пользователя из базы данных
                user_check = await conn.fetchrow("""
                    SELECT login FROM employers WHERE employer_id = $1
                """, int(user_id))
                if user_check:
                    print('[/employer-upload-video] - True User')
                    upload_date_update = await conn.fetchrow("""
                        SELECT upload_date
                        FROM employers_videos
                        WHERE upload_id = $1 AND employer_id = $2
                    """, upload_id_update, int(user_id))
                    moscow_tz = pytz.timezone('Europe/Moscow')
                    upload_date_moscow = upload_date_update['upload_date'].astimezone(moscow_tz)
                    date_name_part = re.sub(r'[^a-zA-Z0-9_-]', '_', str(upload_date_moscow))
                    folder_name = f'uploads_files/employer_{user_id}_{date_name_part}'

                    # Проверка MIME-типа файла
                    file_mime_type = mime.from_buffer(file.file.read(1024))
                    file.file.seek(0)  # Сброс позиции чтения файла

                    if not file_mime_type.startswith('video/'):
                        raise HTTPException(status_code=400, detail={"success": False, "message": "Uploaded file is not a video"})
                    else:
                        file_name = f'employer_{file.filename}'
                        video_path = f"{folder_name}/{file_name}"

                        videos_path_list = await conn.fetchrow("""
                            SELECT videos_url
                            FROM employers_videos
                            WHERE upload_id = $1 AND employer_id = $2
                        """, upload_id_update, int(user_id))
                        if videos_path_list:

                            videos_path_list['videos_url'].append(video_path)
                            videos_path_list_update = videos_path_list['videos_url']

                            if not os.path.exists(video_path):
                                with open(video_path, "wb") as buffer:
                                    shutil.copyfileobj(file.file, buffer)
                                    
                                    print(f'[/employer-upload-video] - Video {file_name} saved')
                            else:
                                print(f'[/employer-upload-video] - Video {file_name} already exists')
                                raise Exception(f'Video {file_name} already exists')

                            await conn.execute("""
                                UPDATE employers_videos
                                SET videos_url = $1
                                WHERE upload_id = $2 AND employer_id = $3
                            """, videos_path_list_update, upload_id_update, int(user_id))
                            print('[/employer-upload-video] Updated videos urls list. List now: ', videos_path_list)
                            return JSONResponse(status_code=200, content={"success": True})
                        else:
                            print('[/employer-upload-video] - False upload_id')
                            raise HTTPException(status_code=404, detail={"success": False, "message": f"upload_id = {upload_id} Not found"})
                else:
                    print('[/employer-upload-video] - False User')
                    raise HTTPException(status_code=404, detail={"success": False, "message": f"User not found"})
            else:
                print('[/employer-upload-video] - False metadata')
                raise "Uploaded file dont have metadata"
    except Exception as e:
        if not isinstance(e, HTTPException):
            print(f'[/employer-upload-video] ERROR: {e}')
            raise HTTPException(status_code=500, detail={"success": False, "message": str(e)})
