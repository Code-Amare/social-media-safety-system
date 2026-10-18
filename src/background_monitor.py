# ============================================================
# EGATE AI - UNIFIED BACKGROUND MONITOR (Windows)
# ============================================================
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Windows: emoji printed into a redirected log file crash without this.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import secrets
import threading
import time
import webbrowser
from pathlib import Path

import joblib
import pandas as pd
import requests
import torch  # noqa: F401
from dotenv import load_dotenv
from PIL import ImageGrab
from transformers.pipelines import pipeline

# ============================================================
# PATHS  (project/src/background_monitor.py)
# ============================================================
SRC_DIR = Path(__file__).resolve().parent  # project/src
ROOT_DIR = SRC_DIR.parent  # project
DATA_DIR = SRC_DIR  # all json/log files live in src/

load_dotenv(SRC_DIR / ".env")
load_dotenv()

MODEL_PATH = ROOT_DIR / "models" / "addiction_pipeline.pkl"

PROFILE_PATH = DATA_DIR / "user_profile.json"
USAGE_PATH = DATA_DIR / "current_usage.json"
TELEGRAM_AUTH_PATH = DATA_DIR / "telegram_authorized.json"
TELEGRAM_CODE_SUBMIT_PATH = DATA_DIR / "telegram_code_submit.json"
TELEGRAM_PAIRING_STATUS_PATH = DATA_DIR / "telegram_pairing_status.json"
PENDING_CODES_PATH = DATA_DIR / "telegram_pending_codes.json"
LOG_FILE = DATA_DIR / "monitor_debug.log"

# ============================================================
# SETTINGS
# ============================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
REGISTRATION_CODE_LIFETIME = 600  # 10 minutes

NSFW_THRESHOLD = 0.10
CHECK_INTERVAL_SECONDS = 3
NSFW_COOLDOWN_SECONDS = 30

ADDICTION_CHECK_INTERVAL = 300
ADDICTION_HIGH_RISK_THRESHOLD = 0.10

LOCKOUT_DURATION = 300
YOUTUBE_URL = "https://www.youtube.com/watch?v=1ZYbU82GVz4"

# ============================================================
# GLOBAL STATE
# ============================================================
authorized_chat_id = None
pending_codes = {}
telegram_lock = threading.Lock()
last_nsfw_alert_time = 0
last_addiction_alert_time = 0
monitor_running = True


# ============================================================
# LOGGING / JSON HELPERS
# ============================================================
def log(message):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass


def write_json_atomic(path, data):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ============================================================
# TELEGRAM AUTH STORAGE
# ============================================================
def load_authorized_chat():
    global authorized_chat_id
    try:
        with open(TELEGRAM_AUTH_PATH, "r", encoding="utf-8") as f:
            authorized_chat_id = json.load(f).get("chat_id")
        if authorized_chat_id:
            log(f"Authorized Telegram account loaded: {authorized_chat_id}")
    except Exception:
        authorized_chat_id = None


def save_authorized_chat(chat_id):
    global authorized_chat_id
    try:
        write_json_atomic(TELEGRAM_AUTH_PATH, {"chat_id": chat_id})
        authorized_chat_id = chat_id
        log(f"Telegram account authorized: {chat_id}")
    except Exception as e:
        log(f"Could not save Telegram authorization: {e}")


# ============================================================
# PENDING CODES (persisted across monitor restarts)
# ============================================================
def save_pending_codes():
    """Call while holding telegram_lock."""
    try:
        write_json_atomic(PENDING_CODES_PATH, pending_codes)
    except Exception as e:
        log(f"Could not save pending codes: {e}")


def load_pending_codes():
    try:
        with open(PENDING_CODES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        now = time.time()
        with telegram_lock:
            for code, info in data.items():
                if info.get("expires", 0) > now:
                    pending_codes[code] = info
    except Exception:
        pass


# ============================================================
# TELEGRAM API
# ============================================================
def telegram_api(method, data=None, timeout=30):
    if not TELEGRAM_TOKEN:
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    try:
        response = requests.post(url, data=data or {}, timeout=timeout)
        return response.json()
    except Exception as e:
        log(f"Telegram API error ({method}): {e}")
        return None


def send_telegram_message(chat_id, message):
    if not chat_id:
        return False
    result = telegram_api("sendMessage", {"chat_id": chat_id, "text": message})
    if result and result.get("ok"):
        log(f"Telegram message sent to {chat_id}")
        return True
    log(f"Telegram send failed: {result}")
    return False


def generate_registration_code():
    return f"{secrets.randbelow(1000000):06d}"


def issue_code(chat_id, first_time):
    code = generate_registration_code()
    expires = time.time() + REGISTRATION_CODE_LIFETIME

    with telegram_lock:
        for old in [c for c, i in pending_codes.items() if i["chat_id"] == chat_id]:
            pending_codes.pop(old, None)
        pending_codes[code] = {"chat_id": chat_id, "expires": expires}
        save_pending_codes()

    intro = "👋 Welcome to Egate AI.\n\n" if first_time else ""
    send_telegram_message(
        chat_id,
        f"{intro}Your computer registration code is:\n\n🔐 {code}\n\n"
        "Enter this 6-digit code on the Egate AI dashboard.\n\n"
        "⏱ The code expires in 10 minutes.",
    )
    log(f"Registration code generated for chat {chat_id}: {code}")


# ============================================================
# TELEGRAM LISTENER
# ============================================================
def parse_command(text):
    if not text or not text.strip():
        return ""
    return text.strip().split()[0].lower().split("@")[0]


def telegram_listener():
    log("Telegram listener started.")

    result = telegram_api("deleteWebhook", {"drop_pending_updates": "false"})
    if result and result.get("ok"):
        log("Telegram webhook disabled.")

    offset = 0

    while monitor_running:
        try:
            result = telegram_api(
                "getUpdates", {"offset": offset, "timeout": 20}, timeout=35
            )

            if not result or not result.get("ok"):
                if result and result.get("error_code") == 409:
                    log(
                        "Telegram 409 conflict: another program is polling "
                        "this bot (another monitor instance?)."
                    )
                time.sleep(3)
                continue

            for update in result.get("result", []):
                offset = update["update_id"] + 1

                message = update.get("message")
                if not message:
                    continue

                if time.time() - message.get("date", 0) > REGISTRATION_CODE_LIFETIME:
                    continue

                chat_id = message.get("chat", {}).get("id")
                if not chat_id:
                    continue

                command = parse_command(message.get("text", ""))

                if command == "/start":
                    issue_code(chat_id, first_time=True)

                elif command == "/newcode":
                    issue_code(chat_id, first_time=False)

                elif command == "/status":
                    if authorized_chat_id == chat_id:
                        send_telegram_message(
                            chat_id, "✅ This Telegram account is paired with Egate AI."
                        )
                    else:
                        send_telegram_message(
                            chat_id, "⚠️ This Telegram account is not paired."
                        )

                elif command == "/test":
                    if authorized_chat_id == chat_id:
                        send_telegram_message(
                            chat_id,
                            "🟢 Egate AI test successful.\n\nTelegram alerts are working.",
                        )
                    else:
                        send_telegram_message(
                            chat_id, "⚠️ This account is not paired with the computer."
                        )

        except Exception as e:
            log(f"Telegram listener error: {e}")
            time.sleep(5)


# ============================================================
# PAIRING VIA DASHBOARD
# ============================================================
def _write_pairing_status(status, message):
    try:
        write_json_atomic(
            TELEGRAM_PAIRING_STATUS_PATH,
            {"status": status, "message": message, "updated_at": time.time()},
        )
    except Exception as e:
        log(f"Could not write pairing status: {e}")


def check_code_submission():
    if not TELEGRAM_CODE_SUBMIT_PATH.exists():
        return

    try:
        with open(TELEGRAM_CODE_SUBMIT_PATH, "r", encoding="utf-8") as f:
            code = str(json.load(f).get("code", "")).strip()
    except Exception as e:
        log(f"Could not read submitted Telegram code: {e}")
        code = ""

    try:
        os.remove(TELEGRAM_CODE_SUBMIT_PATH)
    except Exception:
        pass

    with telegram_lock:
        known = list(pending_codes.keys())
    log(f"Code submitted: '{code}' | pending codes: {known}")

    if len(code) != 6 or not code.isdigit():
        _write_pairing_status("error", "Enter exactly 6 digits.")
        return

    with telegram_lock:
        registration = pending_codes.get(code)

    if not registration:
        _write_pairing_status(
            "error",
            "Invalid registration code. Send /newcode to the bot for a new one.",
        )
        return

    if time.time() > registration["expires"]:
        with telegram_lock:
            pending_codes.pop(code, None)
            save_pending_codes()
        _write_pairing_status(
            "error", "That code expired. Send /newcode to the bot for a new one."
        )
        return

    chat_id = registration["chat_id"]
    save_authorized_chat(chat_id)

    with telegram_lock:
        pending_codes.pop(code, None)
        save_pending_codes()

    send_telegram_message(
        chat_id,
        "✅ COMPUTER PAIRED\n\nThis Telegram account is now connected to "
        "Egate AI.\n\nYou will receive safety alerts from the computer.",
    )
    _write_pairing_status("success", "Computer paired successfully.")
    log(f"Telegram pairing completed via dashboard for chat_id {chat_id}")


def code_submission_watcher():
    log("Telegram code-submission watcher started.")
    while monitor_running:
        try:
            check_code_submission()
        except Exception as e:
            log(f"Code-submission watcher error: {e}")
        time.sleep(0.5)


# ============================================================
# PROFILE / USAGE
# ============================================================
def load_user_profile():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"Missing: {PROFILE_PATH}")
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_usage_data():
    if not USAGE_PATH.exists():
        raise FileNotFoundError(f"Missing: {USAGE_PATH}")
    with open(USAGE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# ADDICTION MODEL
# ============================================================
def load_addiction_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    artifact = joblib.load(MODEL_PATH)
    log(f"Addiction model loaded from {MODEL_PATH}")
    return artifact["preprocessor"], artifact["model"], artifact["feature_columns"]


def build_addiction_features(profile, usage, feature_columns):
    features = {
        "daily_screen_time_hours": usage.get("daily_screen_time_hours", 0.0),
        "social_media_hours": usage.get("social_media_hours", 0.0),
        "gaming_hours": usage.get("gaming_hours", 0.0),
        "notifications_per_day": usage.get("notifications_per_day", 0),
        "app_opens_per_day": usage.get("app_opens_per_day", 0),
        "weekend_screen_time": usage.get(
            "weekend_screen_time", profile.get("weekend_screen_time", 0.0)
        ),
        "age": profile.get("age"),
        "gender": profile.get("gender"),
        "work_study_hours": profile.get("work_study_hours", 0.0),
        "sleep_hours": profile.get("sleep_hours", 7.0),
        "stress_level": profile.get("stress_level", "Medium"),
        "academic_work_impact": profile.get("academic_work_impact", "No"),
    }

    missing = [c for c in feature_columns if features.get(c) is None]
    if missing:
        raise ValueError("Missing model features: " + ", ".join(missing))

    return pd.DataFrame([features], columns=feature_columns)


def predict_addiction(preprocessor, model, feature_columns):
    profile = load_user_profile()
    usage = load_usage_data()
    X = build_addiction_features(profile, usage, feature_columns)
    X_encoded = preprocessor.transform(X)

    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else 1
    probability = float(model.predict_proba(X_encoded)[0][idx])

    return int(probability >= 0.50), probability, X.iloc[0].to_dict()


def send_addiction_alert(probability, features):
    global last_addiction_alert_time

    if not authorized_chat_id:
        log("Addiction alert skipped: no Telegram account paired.")
        return

    now = time.time()
    if now - last_addiction_alert_time < ADDICTION_CHECK_INTERVAL:
        return

    message = (
        "⚠️ EGATE AI USAGE ALERT\n\n"
        f"Estimated model risk: {probability:.1%}\n\n"
        f"Daily screen time: {features['daily_screen_time_hours']:.1f} h\n"
        f"Social media: {features['social_media_hours']:.1f} h\n"
        f"Gaming: {features['gaming_hours']:.1f} h\n"
        f"Work/study: {features['work_study_hours']:.1f} h\n"
        f"Sleep: {features['sleep_hours']:.1f} h\n"
        f"Notifications: {features['notifications_per_day']}\n"
        f"App opens: {features['app_opens_per_day']}\n\n"
        "Please review the usage pattern."
    )

    if send_telegram_message(authorized_chat_id, message):
        last_addiction_alert_time = now


# ============================================================
# NSFW DETECTION
# ============================================================
def load_nsfw_model():
    log("Loading NSFW detection model...")
    classifier = pipeline(
        "image-classification", model="Falconsai/nsfw_image_detection", device=-1
    )
    log("NSFW detection model loaded.")
    return classifier


def get_nsfw_score(classifier, screen):
    for item in classifier(screen):
        if str(item.get("label", "")).lower() == "nsfw":
            return float(item.get("score", 0.0))
    return 0.0


def send_nsfw_alert(score):
    global last_nsfw_alert_time

    now = time.time()
    if now - last_nsfw_alert_time < NSFW_COOLDOWN_SECONDS:
        return

    if not authorized_chat_id:
        log("NSFW detected but no Telegram account is paired.")
        return

    message = (
        "⚠️ SAFETY ALERT\n\n"
        "Potentially inappropriate content was detected on the child's screen.\n\n"
        f"Detection score: {score:.2f}\n\n"
        "A safety intervention has been started."
    )

    if send_telegram_message(authorized_chat_id, message):
        last_nsfw_alert_time = now


def open_youtube():
    log("Opening YouTube intervention video.")
    try:
        webbrowser.open(YOUTUBE_URL)
        log("YouTube opened successfully.")
    except Exception as e:
        log(f"Could not open YouTube: {e}")


def enforce_nsfw_intervention(score):
    log(f"NSFW intervention started. Score={score:.2f}")
    send_nsfw_alert(score)
    open_youtube()

    log("5-minute NSFW intervention cooldown started.")
    end_time = time.time() + LOCKOUT_DURATION
    while time.time() < end_time and monitor_running:
        time.sleep(1)
    log("NSFW intervention cooldown finished.")


# ============================================================
# ADDICTION WORKER
# ============================================================
def addiction_monitor_worker(preprocessor, model, feature_columns):
    while monitor_running:
        try:
            _, probability, features = predict_addiction(
                preprocessor, model, feature_columns
            )
            log(f"Addiction prediction: {probability:.4f}")
            print(f"🧠 Addiction risk: {probability:.1%}", flush=True)

            if probability >= ADDICTION_HIGH_RISK_THRESHOLD:
                send_addiction_alert(probability, features)

        except Exception as e:
            log(f"Addiction monitor error: {e}")
            print(f"⚠️ Addiction monitor: {e}", flush=True)

        for _ in range(ADDICTION_CHECK_INTERVAL):
            if not monitor_running:
                break
            time.sleep(1)


# ============================================================
# MAIN
# ============================================================
def main():
    global monitor_running

    print("=" * 60)
    print("             EGATE AI MONITOR")
    print("=" * 60, flush=True)

    log("=" * 60)
    log("EGATE AI MONITOR STARTING")
    log(f"Data folder: {DATA_DIR}")
    log("=" * 60)

    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN is not set (put it in src/.env).")
        log("TELEGRAM_BOT_TOKEN missing.")
        return

    info = telegram_api("getMe")
    if not info or not info.get("ok"):
        print("❌ Telegram connection failed (check token / internet / firewall).")
        log(f"Telegram connection failed: {info}")
        return

    print(f"✅ Telegram bot: @{info['result'].get('username', 'unknown')}", flush=True)

    load_authorized_chat()
    load_pending_codes()

    threading.Thread(target=telegram_listener, daemon=True).start()
    threading.Thread(target=code_submission_watcher, daemon=True).start()

    if authorized_chat_id:
        print("✅ Existing Telegram pairing found.")
    else:
        print("⚠️ Telegram account is not paired yet. Use the dashboard to pair.")

    print("Loading addiction model...", flush=True)
    try:
        preprocessor, addiction_model, feature_columns = load_addiction_model()
        print("✅ Addiction model loaded.")
        threading.Thread(
            target=addiction_monitor_worker,
            args=(preprocessor, addiction_model, feature_columns),
            daemon=True,
        ).start()
    except Exception as e:
        print(f"⚠️ Addiction model unavailable (continuing): {e}")
        log(f"Addiction model loading error: {e}")

    print("Loading NSFW detection model...", flush=True)
    try:
        nsfw_classifier = load_nsfw_model()
        print("✅ NSFW model loaded.")
    except Exception as e:
        print(f"⚠️ NSFW model failed (Telegram pairing still available): {e}")
        log(f"NSFW model loading error: {e}")
        nsfw_classifier = None

    print("=" * 60)
    print("🟢 EGATE AI IS RUNNING")
    print("=" * 60)
    print("Telegram alerts: " + ("ON" if authorized_chat_id else "OFF (not paired)"))
    print("Press Ctrl+C to stop.", flush=True)
    log("Egate AI monitoring started.")

    try:
        while monitor_running:
            if nsfw_classifier is not None:
                try:
                    screen = ImageGrab.grab().convert("RGB")
                    nsfw_score = get_nsfw_score(nsfw_classifier, screen)
                    log(f"Screen scanned. NSFW score={nsfw_score:.4f}")

                    if nsfw_score >= NSFW_THRESHOLD:
                        print(f"⚠️ NSFW detected: {nsfw_score:.2f}", flush=True)
                        enforce_nsfw_intervention(nsfw_score)
                except Exception as e:
                    log(f"Screen scan error: {e}")

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("🛑 Stopping Egate AI...")
        log("--- Egate AI stopped by user ---")
    except Exception as e:
        print(f"❌ Main monitor error: {e}")
        log(f"Main monitor error: {e}")
    finally:
        monitor_running = False

    print("✅ Egate AI stopped.")


if __name__ == "__main__":
    main()
