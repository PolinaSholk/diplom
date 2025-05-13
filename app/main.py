import asyncio
import os
from datetime import timedelta, datetime

import pandas as pd
import pytz
from catboost import CatBoostRegressor
from fastapi import Depends, HTTPException, status
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from starlette.responses import RedirectResponse, HTMLResponse

from .auth import authenticate_user, create_access_token, get_password_hash
from .auth import get_current_user
from .config import ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY, ALGORITHM
from .database import SessionLocal
from .database import engine
from .models import Base, UploadedFile
from .models import User
from .schemas import User, Token

app = FastAPI()

# Подключение статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Создание таблиц в БД
Base.metadata.drop_all(bind=engine)  # Удаляем старые таблицы
Base.metadata.create_all(bind=engine)  # Создаем новые

model = None
last_prediction_time = None
predictions_cache = {}


def train_model(X_train, y_train, X_test, y_test):
    """Обучение CatBoost модели"""
    global model
    cat_features = ['location_type', 'weather', 'holiday', 'is_weekend']
    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_strength=0.5,  # Уменьшает переобучение
        grow_policy='Lossguide',
        loss_function='MAE',
        cat_features=cat_features,
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )
    return model


async def update_predictions():
    """Обновление прогнозов каждый час"""
    global last_prediction_time, predictions_cache
    while True:
        now = datetime.now(pytz.utc)
        if last_prediction_time is None or (now - last_prediction_time).total_seconds() >= 3600:
            db = SessionLocal()
            try:
                file_record = db.query(UploadedFile).order_by(UploadedFile.id.desc()).first()
                if file_record:
                    df = pd.read_csv(file_record.filepath, sep=';')

                    if 'value' in df.columns and 'timestamp' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M')

                        # Добавляем фичи
                        df['hour'] = df['datetime'].dt.hour
                        df['day_of_week'] = df['datetime'].dt.weekday
                        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                        df['month'] = df['datetime'].dt.month

                        # Целевая переменная - абсолютная сумма операций (для прогнозирования нагрузки)
                        df['target'] = df['amount'].abs()

                        # Категориальные фичи
                        cat_features = ['location_type', 'weather', 'holiday', 'is_weekend']

                        # Фичи для модели
                        features = ['hour', 'day_of_week', 'is_weekend', 'month',
                                    'day_of_year', 'hour_sin', 'hour_cos',
                                    'location_type', 'weather', 'holiday']

                        # Фильтр данных (например, для одного банкомата)
                        atm_data = df.copy()

                        # Сортируем по времени
                        atm_data = atm_data.sort_values('datetime')

                        # Разделение на train/test (80/20 с сохранением временного порядка)
                        train_size = int(len(atm_data) * 0.8)
                        train = atm_data.iloc[:train_size]
                        test = atm_data.iloc[train_size:]

                        X_train, y_train = train[features], train['target']
                        X_test, y_test = test[features], test['target']

                        # Обучение модели
                        model = train_model(X_train, y_train, X_test, y_test)

                        # Прогнозирование
                        last_time = df['datetime'].max()
                        next_1h = last_time + pd.Timedelta(hours=1)
                        next_2h = last_time + pd.Timedelta(hours=1)

                        # Особенности для прогноза
                        features_1h = [
                            (next_1h - df['datetime'].min()).total_seconds(),
                            next_1h.hour,
                            next_1h.dayofweek
                        ]

                        features_2h = [
                            (next_2h - df['datetime'].min()).total_seconds(),
                            next_2h.hour,
                            next_2h.dayofweek
                        ]

                        pred_1h = model.predict([features_1h])[0]
                        pred_2h = model.predict([features_2h])[0]

                        # Определение необходимости обслуживания
                        maintenance_1h = pred_1h < 1000 or pred_1h > 9000  # Примерные границы
                        maintenance_2h = pred_2h < 1000 or pred_2h > 9000

                        predictions_cache = {
                            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'pred_1h': pred_1h,
                            'pred_2h': pred_2h,
                            'maintenance_1h': maintenance_1h,
                            'maintenance_2h': maintenance_2h
                        }

                        last_prediction_time = now
            except Exception as e:
                print(f"Ошибка при обновлении прогнозов: {e}")
            finally:
                db.close()

        await asyncio.sleep(3600)

@app.on_event("startup")
async def startup_event():
    """Запуск фоновой задачи при старте сервера"""
    asyncio.create_task(update_predictions())

# Маршруты
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login_for_access_token(
        request: Request,
        username: str = Form(...),
        password: str = Form(...)
):
    db = SessionLocal()
    user = authenticate_user(db, username, password)
    db.close()

    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Неверное имя пользователя или пароль"},
            status_code=status.HTTP_401_UNAUTHORIZED
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=1800,  # 30 минут
        secure=False,  # Для разработки, в production должно быть True
        samesite="lax"
    )
    return response


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def register_user(request: Request, username: str = Form(...),
        password: str = Form(...), password_confirm: str = Form(...), email: str = Form(...)):
    if password != password_confirm:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Пароли не совпадают"}
        )

    db = SessionLocal()

    # Проверка существования пользователя
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        db.close()
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Пользователь с таким именем уже существует"}
        )

    # Создание нового пользователя
    hashed_password = get_password_hash(password)
    db_user = User(username=username, hashed_password=hashed_password, email=email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    db.close()

    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response


# Защищенные маршруты
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "username": current_user.username}
    )

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "username": current_user.username}
    )


@app.post("/check_files")
async def check_files_integrity(
        current_user: User = Depends(get_current_user)
):
    db = SessionLocal()
    try:
        files = db.query(UploadedFile) \
            .filter(UploadedFile.owner_id == current_user.id) \
            .all()

        results = []
        for file in files:
            exists = os.path.exists(file.filepath)
            results.append({
                "id": file.id,
                "filename": file.filename,
                "db_path": file.filepath,
                "exists": exists,
                "size": os.path.getsize(file.filepath) if exists else 0
            })

        return {"files": results}
    finally:
        db.close()

@app.get("/predictions/{file_id}")
async def show_predictions(
        request: Request,
        file_id: int,
        current_user: User = Depends(get_current_user)
):
    db = SessionLocal()
    try:
        # Получаем файл с проверкой владельца
        file_record = db.query(UploadedFile) \
            .filter(UploadedFile.id == file_id) \
            .filter(UploadedFile.owner_id == current_user.id) \
            .first()

        if not file_record:
            raise HTTPException(
                status_code=404,
                detail="Файл не найден или нет доступа"
            )

        # Проверяем существование файла на диске
        if not os.path.exists(file_record.filepath):
            # Пытаемся восстановить
            if os.path.exists(os.path.join(UPLOAD_DIRECTORY, file_record.filename)):
                file_record.filepath = os.path.join(UPLOAD_DIRECTORY, file_record.filename)
                db.commit()
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Файл данных не найден на сервере"
                )

        # Чтение и обработка файла
        try:
            df = pd.read_csv(file_record.filepath, sep=';')

        # Генерация прогноза (ваш существующий код)

            df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M')

            # Добавляем фичи
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.weekday
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['month'] = df['datetime'].dt.month

            # Целевая переменная - абсолютная сумма операций (для прогнозирования нагрузки)
            df['target'] = df['amount'].abs()

            # Категориальные фичи
            cat_features = ['location_type', 'weather', 'holiday', 'is_weekend']

            # Фичи для модели
            features = ['hour', 'day_of_week', 'is_weekend', 'month',
                        'day_of_year', 'hour_sin', 'hour_cos',
                        'location_type', 'weather', 'holiday']

            # Фильтр данных (например, для одного банкомата)
            atm_data = df.copy()

            # Сортируем по времени
            atm_data = atm_data.sort_values('datetime')

            # Разделение на train/test (80/20 с сохранением временного порядка)
            train_size = int(len(atm_data) * 0.8)
            train = atm_data.iloc[:train_size]
            test = atm_data.iloc[train_size:]

            X_train, y_train = train[features], train['target']
            X_test, y_test = test[features], test['target']

            # Обучаем модель CatBoost
            cat_features = ['location_type', 'weather', 'holiday', 'is_weekend']
            model = CatBoostRegressor(
                iterations=1000,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_strength=0.5,  # Уменьшает переобучение
                grow_policy='Lossguide',
                loss_function='MAE',
                cat_features=cat_features,
                random_seed=42,
                verbose=100
            )
            model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=50
            )

            # Получаем последнюю временную метку
            last_time = df['datetime'].max()

            # Генерируем прогноз на 1 и 2 часа вперед
            predictions = []
            for hours in [1, 2]:
                pred_time = last_time + pd.Timedelta(hours=hours)

                # Создаем признаки для прогноза
                features = [
                    (pred_time - df['datetime'].min()).dt.total_seconds(),
                    pred_time.hour,
                    pred_time.dayofweek
                ]

                # Делаем прогноз
                prediction = model.predict([features])[0]

                # Определяем необходимость обслуживания
                needs_maintenance = prediction < 1000 or prediction > 9000  # Настройте пороги

                predictions.append({
                    'period': f'Через {hours} час(а)',
                    'time': pred_time.strftime('%Y-%m-%d %H:%M'),
                    'prediction': round(prediction, 2),
                    'needs_maintenance': needs_maintenance
                })

            return templates.TemplateResponse(
                "predictions.html",
                {
                    "request": request,
                    "predictions": predictions,
                    "filename": file_record.filename,
                    "file_id": file_id
                })
        except pd.errors.EmptyDataError:
            raise HTTPException(
                status_code=400,
                detail="Файл пуст или имеет неверный формат"
            )
    except Exception as e:
        return templates.TemplateResponse(
            "predictions.html",
            {"request": request, "error": str(e)}
        )
    finally:
        db.close()

UPLOAD_DIRECTORY = "app/uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@app.post("/uploadfile/")
async def create_upload_file(
        request: Request,
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user)
):
    try:
        # Читаем файл в pandas
        df = pd.read_csv(file.file, sep=';')

        # Проверяем необходимые колонки
        if not all(col in df.columns for col in ['atmdeviceid', 'date', 'totalamount']):
            raise ValueError("Файл должен содержать колонки: atmdeviceid, timestamp, amount")

        # Преобразуем timestamp
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Группируем по банкоматам и получаем последние значения
        predictions = []
        for atm_id, group in df.groupby('atmdeviceid'):
            last_record = group.iloc[-1]
            last_amount = last_record['totalamount']
            last_time = last_record['date']

            # Простое прогнозирование (замените на CatBoost при необходимости)
            # Здесь можно добавить вашу модель прогнозирования
            pred1 = round(last_amount * 0.95, 2)  # Пример: уменьшение на 5%
            pred2 = round(last_amount * 0.90, 2)  # Пример: уменьшение на 10%

            predictions.append({
                'atm_id': atm_id,
                'amount_now': last_amount,
                'pred1': pred1,
                'pred2': pred2,
                'needs_collector': last_amount > 10000
            })

        # Создаем HTML таблицу
        table_html = "<table class='table table-striped'><thead><tr>"
        table_html += "<th>atm_id</th><th>amount_now</th><th>pred1 (1h)</th><th>pred2 (2h)</th><th>collector</th>"
        table_html += "</tr></thead><tbody>"

        for pred in predictions:
            table_html += f"<tr><td>{pred['atm_id']}</td>"
            table_html += f"<td>{pred['amount_now']}</td>"
            table_html += f"<td>{pred['pred1']}</td>"
            table_html += f"<td>{pred['pred2']}</td>"
            if pred['needs_collector']:
                table_html += "<td><button class='btn btn-sm btn-success' onclick='scheduleCollection(\"" + str(
                    pred['atm_id']) + "\")'>Вызвать инкассатора</button></td>"
            else:
                table_html += "<td><button class='btn btn-sm btn-secondary' disabled>Не требуется</button></td>"
            table_html += "</tr>"

        table_html += "</tbody></table>"

        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "filename": file.filename,
                "table_html": table_html,
                "message": "Файл успешно обработан"
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "error": f"Ошибка обработки файла: {str(e)}"
            }
        )


@app.get("/select_file")
async def select_file(
        request: Request,
        current_user: User = Depends(get_current_user)
):
    db = SessionLocal()
    files = db.query(UploadedFile) \
        .filter(UploadedFile.owner_id == current_user.id) \
        .order_by(UploadedFile.uploaded_at.desc()) \
        .all()
    db.close()

    return templates.TemplateResponse(
        "select_file.html",
        {"request": request, "files": files}
    )

@app.get("/get_predictions/")
async def get_predictions(
        request: Request,
        current_user: User = Depends(get_current_user)
):
    db = SessionLocal()

    try:
        # Получаем последний загруженный файл пользователя
        file_record = db.query(UploadedFile) \
            .filter(UploadedFile.owner_id == current_user.id) \
            .order_by(UploadedFile.id.desc()) \
            .first()

        if not file_record:
            return templates.TemplateResponse(
                "predictions.html",
                {"request": request, "error": "Нет загруженных данных для прогнозирования"}
            )

        # Проверяем существование файла
        if not os.path.exists(file_record.filepath):
            return templates.TemplateResponse(
                "predictions.html",
                {"request": request, "error": "Файл данных не найден"}
            )

        # Читаем данные из файла
        df = pd.read_csv(file_record.filepath, sep=';')

        df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M')

        # Добавляем фичи
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.weekday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['datetime'].dt.month

        # Целевая переменная - абсолютная сумма операций (для прогнозирования нагрузки)
        df['target'] = df['amount'].abs()

        # Категориальные фичи
        cat_features = ['location_type', 'weather', 'holiday', 'is_weekend']

        # Фичи для модели
        features = ['hour', 'day_of_week', 'is_weekend', 'month',
                    'day_of_year', 'hour_sin', 'hour_cos',
                    'location_type', 'weather', 'holiday']

        # Фильтр данных (например, для одного банкомата)
        atm_data = df.copy()

        # Сортируем по времени
        atm_data = atm_data.sort_values('datetime')

        # Разделение на train/test (80/20 с сохранением временного порядка)
        train_size = int(len(atm_data) * 0.8)
        train = atm_data.iloc[:train_size]
        test = atm_data.iloc[train_size:]

        X_train, y_train = train[features], train['target']
        X_test, y_test = test[features], test['target']

        # Обучаем модель CatBoost
        cat_features = ['location_type', 'weather', 'holiday', 'is_weekend']
        model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_strength=0.5,  # Уменьшает переобучение
            grow_policy='Lossguide',
            loss_function='MAE',
            cat_features=cat_features,
            random_seed=42,
            verbose=100
        )
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50
        )

        # Получаем последнюю временную метку
        last_time = df['datetime'].max()

        # Генерируем прогноз на 1 и 2 часа вперед
        predictions = []
        for hours in [1, 2]:
            pred_time = last_time + pd.Timedelta(hours=hours)

            # Создаем признаки для прогноза
            features = [
                (pred_time - df['datetime'].min()).dt.total_seconds(),
                pred_time.hour,
                pred_time.dayofweek
            ]

            # Делаем прогноз
            prediction = model.predict([features])[0]

            # Определяем необходимость обслуживания
            needs_maintenance = prediction < 1000 or prediction > 9000  # Настройте пороги

            predictions.append({
                'period': f'Через {hours} час(а)',
                'time': pred_time.strftime('%Y-%m-%d %H:%M'),
                'prediction': round(prediction, 2),
                'needs_maintenance': needs_maintenance
            })

        return templates.TemplateResponse(
            "predictions.html",
            {
                "request": request,
                "predictions": predictions,
                "last_update": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "predictions.html",
            {
                "request": request,
                "error": f"Ошибка при генерации прогноза: {str(e)}"
            }
        )
    finally:
        db.close()


@app.post("/token", response_model=Token)
async def login_for_access_token(username: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    user = authenticate_user(db, username, password)
    db.close()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Разрешаем доступ к статическим файлам и страницам авторизации
    if request.url.path.startswith("/static") or request.url.path in ["/login", "/register"]:
        return await call_next(request)

    # Проверяем токен в cookies
    token = request.cookies.get("access_token")
    if not token:
        if request.url.path != "/":
            return RedirectResponse(url="/login")
        return await call_next(request)

    try:
        # Проверяем валидность токена
        payload = jwt.decode(
            token.replace("Bearer ", ""),
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        request.state.user = payload.get("sub")
    except JWTError:
        response = RedirectResponse(url="/login")
        response.delete_cookie("access_token")
        return response

    return await call_next(request)


@app.post("/schedule_collection/")
async def schedule_collection(
        request: Request,
        current_user: User = Depends(get_current_user)
):
    data = await request.json()
    atm_id = data.get('atm_id')

    # Здесь можно сохранить запрос на инкассацию в базу данных
    print(f"Запланирована инкассация для банкомата {atm_id}")

    return {"success": True, "message": f"Инкассация для {atm_id} запланирована"}

@app.post("/schedule_maintenance/")
async def schedule_maintenance(
        request: Request,
        current_user: User = Depends(get_current_user)
):
    data = await request.json()
    period = data.get('period')

    # Здесь можно сохранить запрос на обслуживание в базу данных
    # Например:
    # db = SessionLocal()
    # maintenance = MaintenanceSchedule(...)
    # db.add(maintenance)
    # db.commit()

    return {"success": True, "message": f"Обслуживание запланировано на {period}"}