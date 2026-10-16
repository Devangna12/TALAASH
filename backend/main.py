"""
TALAASH backend: authentication, case management REST API, and live
camera matching pushed over WebSocket.

Run with:
    uvicorn backend.main:app
"""
import asyncio
import json
import os
import uuid
from datetime import datetime

import cv2
import insightface
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from backend.geocoding import geocode_location

from backend.auth import create_token, decode_token, get_current_user, require_admin
from backend.otp import generate_and_store_otp, send_otp_email, verify_otp
from database import (
    add_case,
    analytics_stats,
    add_case_note,
    add_detection,
    add_report,
    case_stats,
    create_user,
    get_case,
    get_user_by_username,
    init_db,
    list_case_notes,
    list_cases,
    list_cases_missing_coordinates,
    list_detections,
    list_reports,
    list_users,
    load_or_create_index,
    search,
    update_case_coordinates,
    update_case_status,
    update_detection_status,
    update_report_status,
    update_user_password,
    update_user_profile,
)
from security import hash_password, verify_password

app = FastAPI(title="TALAASH")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/report_photos", exist_ok=True)
app.mount("/media", StaticFiles(directory="data"), name="media")

MATCH_THRESHOLD = 0.25
CAMERA_SOURCE = 0  # replace with an RTSP URL for a real CCTV feed

CAMERA_LAT = float(os.environ.get("CAMERA_LAT", "26.9124"))
CAMERA_LNG = float(os.environ.get("CAMERA_LNG", "75.7873"))
DEBUG = True

detector = YOLO("yolov8n-face.pt")

RECOGNITION_MODEL_PATH = os.path.expanduser(
    "~/.insightface/models/buffalo_l/w600k_r50.onnx"
)
recognizer = insightface.model_zoo.get_model(RECOGNITION_MODEL_PATH)
recognizer.prepare(ctx_id=-1)  # ctx_id=-1 runs on CPU

# Shared by both the background camera loop and the REST endpoints,
# so a case opened through the API is immediately matchable by the
# live feed without restarting the server.
db_conn = init_db()
faiss_index = load_or_create_index()

# Updated every frame by camera_loop() and read by the /api/video_feed
# endpoint. A plain module-level variable is enough here -- there's
# only one asyncio event loop and no real concurrent writers, so no
# lock is needed.
latest_jpeg_frame: bytes | None = None


class ConnectionManager:
    def __init__(self):
        self.active = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, payload: dict):
        message = json.dumps(payload)
        for ws in list(self.active):
            try:
                await ws.send_text(message)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


def get_embedding(face_crop):
    face_crop = cv2.resize(face_crop, (112, 112))
    return recognizer.get_feat(face_crop)


def detect_largest_face(image):
    results = detector(image, verbose=False, device="cpu")[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None

    def area(box):
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)

    best = max(boxes, key=area)
    x1, y1, x2, y2 = best.astype(int)
    crop = image[max(0, y1):y2, max(0, x1):x2]
    return crop if crop.size > 0 else None


def open_camera(source):
    if isinstance(source, int):
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


async def camera_loop():
    cap = open_camera(CAMERA_SOURCE)

    if not cap.isOpened():
        print(f"ERROR: could not open camera source {CAMERA_SOURCE}.")
        return

    print("Camera opened successfully. Starting detection loop...")
    frame_count = 0
    consecutive_failures = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            consecutive_failures += 1
            if consecutive_failures == 1:
                print("Warning: dropped a frame, retrying...")
            elif consecutive_failures == 50:
                print("ERROR: camera stopped responding.")
                return
            await asyncio.sleep(0.2)
            continue

        consecutive_failures = 0
        frame_count += 1

        detections = detector(frame, verbose=False, device="cpu")[0]
        num_faces = len(detections.boxes)

        if DEBUG and frame_count % 30 == 0:
            print(f"[debug] frame {frame_count}: {num_faces} face(s) detected")

        for box in detections.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box.astype(int)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            embedding = get_embedding(crop)
            matches = search(faiss_index, embedding, top_k=1)

            if DEBUG and matches:
                closest_id, closest_score = matches[0]
                closest_case = get_case(db_conn, closest_id)
                closest_name = closest_case["name"] if closest_case else "?"
                print(f"[debug] closest case: {closest_name} - confidence {closest_score:.3f}")

            is_match = bool(matches) and matches[0][1] >= MATCH_THRESHOLD
            box_color = (0, 0, 255) if is_match else (0, 200, 0)  # BGR: red / green
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            if not is_match:
                continue

            case_id, score = matches[0]
            case = get_case(db_conn, case_id)
            if case is None or case["status"] == "Closed":
                continue

            cv2.putText(
                frame, case["name"], (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2,
            )

            detection_id = add_detection(db_conn, case_id, "Cam 01", score)

            print(f"MATCH: case {case['case_number']} ({case['name']}) - confidence {score:.3f}")
            await manager.broadcast(
                {
                    "type": "possible_match",
                    "detection_id": detection_id,
                    "case_number": case["case_number"],
                    "name": case["name"],
                    "status": case["status"],
                    "camera": "Cam 01",
                    "confidence": round(score * 100, 1),
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
            )

        global latest_jpeg_frame
        ok_encode, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok_encode:
            latest_jpeg_frame = buffer.tobytes()

        await asyncio.sleep(0.03)


@app.on_event("startup")
async def start_camera_task():
    asyncio.create_task(camera_loop())


@app.websocket("/ws/alerts")
async def alerts_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        if latest_jpeg_frame is not None:
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + latest_jpeg_frame + b"\r\n"
            )
        # Slightly slower than the capture loop so this never
        # out-paces new frames being produced.
        await asyncio.sleep(0.05)


@app.get("/api/video_feed")
async def video_feed(token: str):
    # An <img> tag can't send an Authorization header, so the token
    # is passed as a query parameter instead and verified the same
    # way get_current_user() would.
    decode_token(token)
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --------------------------------------------------------------- auth

@app.post("/api/auth/login")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_username(db_conn, form.username)
    if user is None or not verify_password(form.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    code = generate_and_store_otp(user["username"], purpose="login")
    send_otp_email(user.get("email"), user["username"], code, purpose="login")

    # No access token yet -- the frontend must call /api/auth/verify-otp
    # with the code before it receives a real, usable token.
    return {
        "otp_required": True,
        "username": user["username"],
        "message": "A verification code has been sent to your registered email.",
    }


@app.post("/api/auth/verify-otp")
async def verify_otp_endpoint(username: str = Form(...), otp: str = Form(...)):
    user = get_user_by_username(db_conn, username)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid session, please log in again.")

    if not verify_otp(username, otp, purpose="login"):
        raise HTTPException(status_code=401, detail="Incorrect or expired verification code.")

    token = create_token(user["username"], user["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"],
        "full_name": user["full_name"],
    }


@app.post("/api/auth/forgot-password")
async def forgot_password(username: str = Form(...)):
    user = get_user_by_username(db_conn, username)
    # Always return the same response whether or not the username
    # exists, so this endpoint can't be used to check which usernames
    # are registered. The email (or console line) only actually goes
    # out if the account is real.
    if user is not None:
        code = generate_and_store_otp(user["username"], purpose="reset")
        send_otp_email(user.get("email"), user["username"], code, purpose="reset")

    return {
        "message": "If that username exists, a password reset code has been sent to its registered email.",
    }


@app.post("/api/auth/reset-password")
async def reset_password(
    username: str = Form(...),
    otp: str = Form(...),
    new_password: str = Form(...),
):
    user = get_user_by_username(db_conn, username)
    if user is None:
        raise HTTPException(status_code=400, detail="Incorrect or expired reset code.")

    if not verify_otp(username, otp, purpose="reset"):
        raise HTTPException(status_code=400, detail="Incorrect or expired reset code.")

    if not new_password:
        raise HTTPException(status_code=400, detail="Password cannot be empty.")

    update_user_password(db_conn, username, hash_password(new_password))
    return {"message": "Password updated. You can now log in with your new password."}


@app.patch("/api/auth/profile")
async def update_profile(
    full_name: str | None = Form(None),
    email: str | None = Form(None),
    new_password: str | None = Form(None),
    current_user: dict = Depends(get_current_user),
):
    username = current_user["username"]
    update_user_profile(db_conn, username, full_name=full_name, email=email)

    if new_password:
        update_user_password(db_conn, username, hash_password(new_password))

    return get_user_by_username(db_conn, username)


@app.get("/api/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    user = get_user_by_username(db_conn, current_user["username"])
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    # Never send the password hash to the frontend.
    return {k: v for k, v in user.items() if k != "password_hash"}


@app.get("/api/auth/users")
async def api_list_users(admin: dict = Depends(require_admin)):
    return list_users(db_conn)


@app.post("/api/auth/users")
async def api_create_user(
    username: str = Form(...),
    password: str = Form(...),
    full_name: str | None = Form(None),
    email: str | None = Form(None),
    role: str = Form("officer"),
    admin: dict = Depends(require_admin),
):
    if get_user_by_username(db_conn, username):
        raise HTTPException(status_code=400, detail="That username is already taken.")
    try:
        create_user(db_conn, username, hash_password(password), role=role, full_name=full_name, email=email)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"username": username, "role": role}


# -------------------------------------------------------------- cases
# All case endpoints require a logged-in user (admin or officer).

@app.get("/api/cases")
async def api_list_cases(status: str | None = None, current_user: dict = Depends(get_current_user)):
    return list_cases(db_conn, status=status)


@app.get("/api/cases/{case_id}")
async def api_get_case(case_id: int, current_user: dict = Depends(get_current_user)):
    case = get_case(db_conn, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    case["notes"] = list_case_notes(db_conn, case_id)
    return case


@app.post("/api/cases")
async def api_create_case(
    name: str = Form(...),
    age: int | None = Form(None),
    gender: str | None = Form(None),
    contact: str | None = Form(None),
    last_known_location: str | None = Form(None),
    description: str | None = Form(None),
    photo: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    file_bytes = await photo.read()
    np_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    face_crop = detect_largest_face(image)
    if face_crop is None:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded photo.")

    embedding = get_embedding(face_crop)
    lat, lng = geocode_location(last_known_location)
    case_id, case_number = add_case(
        db_conn, faiss_index, name, embedding,
        age=age, gender=gender, contact=contact,
        last_known_location=last_known_location,
        description=description, photo_path=None,
        latitude=lat, longitude=lng,
    )

    return {"id": case_id, "case_number": case_number, "name": name}


@app.patch("/api/cases/{case_id}/status")
async def api_update_status(case_id: int, status: str = Form(...), current_user: dict = Depends(get_current_user)):
    case = get_case(db_conn, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    try:
        update_case_status(db_conn, case_id, status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return get_case(db_conn, case_id)


@app.post("/api/cases/{case_id}/notes")
async def api_add_note(case_id: int, note: str = Form(...), current_user: dict = Depends(get_current_user)):
    case = get_case(db_conn, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    add_case_note(db_conn, case_id, note)
    return {"notes": list_case_notes(db_conn, case_id)}


# --------------------------------------------------------- stats & detections

@app.get("/api/stats/home")
async def api_home_stats(current_user: dict = Depends(get_current_user)):
    stats = case_stats(db_conn)
    stats["cameras_online"] = 1  # single local camera for now
    return stats


@app.get("/api/stats/analytics")
async def api_analytics(days: int = 7, admin: dict = Depends(require_admin)):
    # Clamp so a request can't ask for an unreasonable window.
    days = max(1, min(days, 90))
    return analytics_stats(db_conn, days=days)


@app.get("/api/detections")
async def api_list_detections(period: str | None = None, limit: int | None = None,
                                current_user: dict = Depends(get_current_user)):
    return list_detections(db_conn, period=period, limit=limit)


@app.patch("/api/detections/{detection_id}/status")
async def api_update_detection_status(detection_id: int, status: str = Form(...),
                                        current_user: dict = Depends(get_current_user)):
    try:
        update_detection_status(db_conn, detection_id, status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": detection_id, "status": status}


@app.get("/api/map/data")
async def api_map_data(current_user: dict = Depends(get_current_user)):
    all_cases = list_cases(db_conn)
    geocoded_cases = [
        c for c in all_cases
        if c["latitude"] is not None and c["longitude"] is not None
    ]

    recent = list_detections(db_conn, period="today")

    return {
        "cases": geocoded_cases,
        "cameras": [{"name": "Cam 01", "latitude": CAMERA_LAT, "longitude": CAMERA_LNG}],
        "recent_detections": [
            {**d, "latitude": CAMERA_LAT, "longitude": CAMERA_LNG}
            for d in recent
        ],
    }


# -------------------------------------------------------------- Reports

@app.post("/api/reports")
async def api_create_report(
    case_id: int | None = Form(None),
    location: str | None = Form(None),
    description: str | None = Form(None),
    photo: UploadFile | None = File(None),
    current_user: dict = Depends(get_current_user),
):
    photo_path = None
    if photo is not None:
        ext = os.path.splitext(photo.filename or "")[1] or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        photo_path = f"report_photos/{filename}"
        with open(f"data/{photo_path}", "wb") as f:
            f.write(await photo.read())

    report_id = add_report(
        db_conn, case_id, current_user["username"],
        location=location, description=description, photo_path=photo_path,
    )
    return {"id": report_id}


@app.get("/api/reports")
async def api_list_reports(mine: bool = False, current_user: dict = Depends(get_current_user)):
    officer_filter = current_user["username"] if mine else None
    return list_reports(db_conn, officer_username=officer_filter)


@app.patch("/api/reports/{report_id}/status")
async def api_update_report_status(report_id: int, status: str = Form(...),
                                     current_user: dict = Depends(get_current_user)):
    try:
        update_report_status(db_conn, report_id, status)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": report_id, "status": status}


@app.post("/api/map/backfill-geocoding")
async def api_backfill_geocoding(admin: dict = Depends(require_admin)):
    """Re-attempts geocoding for any case that has a location on file
    but no coordinates yet -- useful after adding GOOGLE_MAPS_SERVER_KEY
    for the first time, so older cases aren't stuck off the map."""
    pending = list_cases_missing_coordinates(db_conn)
    updated = 0
    for case in pending:
        lat, lng = geocode_location(case["last_known_location"])
        if lat is not None and lng is not None:
            update_case_coordinates(db_conn, case["id"], lat, lng)
            updated += 1

    return {"checked": len(pending), "updated": updated}