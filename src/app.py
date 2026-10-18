import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import psutil
import requests
import streamlit as st
from dotenv import load_dotenv

# ============================================================
# PATHS  (project/src/app.py)
# ============================================================
SRC_DIR = Path(__file__).resolve().parent  # project/src
ROOT_DIR = SRC_DIR.parent  # project
DATA_DIR = SRC_DIR  # same folder as the monitor

load_dotenv(SRC_DIR / ".env")
load_dotenv()

MODEL_PATH = ROOT_DIR / "models" / "addiction_pipeline.pkl"

PROFILE_PATH = DATA_DIR / "user_profile.json"
PROFILES_PATH = DATA_DIR / "user_profiles.json"
USAGE_PATH = DATA_DIR / "current_usage.json"
USAGES_PATH = DATA_DIR / "user_usage.json"
ACTIVE_PROFILE_PATH = DATA_DIR / "active_profile.json"
TELEGRAM_AUTH_PATH = DATA_DIR / "telegram_authorized.json"
TELEGRAM_CODE_SUBMIT_PATH = DATA_DIR / "telegram_code_submit.json"
TELEGRAM_PAIRING_STATUS_PATH = DATA_DIR / "telegram_pairing_status.json"
LOG_FILE = DATA_DIR / "monitor_debug.log"
MONITOR_PATH = SRC_DIR / "background_monitor.py"
MONITOR_SCRIPT_NAME = MONITOR_PATH.name
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="NIP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
    .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        background: linear-gradient(135deg, rgba(52,152,219,0.16), rgba(155,89,182,0.12));
        border: 1px solid rgba(127,127,127,0.20); margin-bottom: 1rem;
    }
    .hero-title { font-size: 2.25rem; font-weight: 800; margin-bottom: 0.2rem; }
    .hero-subtitle { opacity: 0.75; font-size: 1rem; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(127,127,127,0.16); border-radius: 12px; padding: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# JSON AND PROFILE HELPERS
# ============================================================
def load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    os.replace(temp_path, path)


def normalize_profile(profile):
    profile = profile if isinstance(profile, dict) else {}
    return {
        "age": int(profile.get("age", 18)),
        "gender": profile.get("gender", "Male"),
        "work_study_hours": float(profile.get("work_study_hours", 4.0)),
        "sleep_hours": float(profile.get("sleep_hours", 7.0)),
        "stress_level": profile.get("stress_level", "Medium"),
        "academic_work_impact": profile.get("academic_work_impact", "No"),
        "weekend_screen_time": float(profile.get("weekend_screen_time", 0.0)),
    }


def load_profiles():
    data = load_json(PROFILES_PATH)
    if isinstance(data, dict):
        return {
            str(name): normalize_profile(profile)
            for name, profile in data.items()
            if isinstance(profile, dict)
        }

    legacy = load_json(PROFILE_PATH)
    if isinstance(legacy, dict):
        profiles = {"Default User": normalize_profile(legacy)}
        save_json(PROFILES_PATH, profiles)
        return profiles

    return {}


def save_profiles(profiles):
    save_json(PROFILES_PATH, profiles)


def load_user_usages():
    data = load_json(USAGES_PATH)
    return data if isinstance(data, dict) else {}


def save_user_usages(usages):
    save_json(USAGES_PATH, usages)


def get_active_profile_name():
    data = load_json(ACTIVE_PROFILE_PATH)
    return data.get("name") if isinstance(data, dict) else None


def set_active_profile_name(name):
    save_json(ACTIVE_PROFILE_PATH, {"name": name})


def activate_profile(name, profile):
    """Keep the legacy profile file synchronized for the monitor."""
    save_json(PROFILE_PATH, normalize_profile(profile))


def get_selected_usage(name):
    usage = load_user_usages().get(name)
    if isinstance(usage, dict):
        return usage
    legacy = load_json(USAGE_PATH)
    return legacy if isinstance(legacy, dict) else None


def save_selected_usage(name, usage):
    usages = load_user_usages()
    usages[name] = usage
    save_user_usages(usages)
    save_json(USAGE_PATH, usage)


# ============================================================
# MODEL LOADING AND PREDICTION
# ============================================================
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    if not MODEL_PATH.is_file():
        return None
    # Only load pickle/joblib files you trust.
    return joblib.load(MODEL_PATH)


def get_model_artifact():
    try:
        return load_model(), None
    except Exception as exc:
        return None, str(exc)


def run_prediction(profile, usage):
    if not MODEL_PATH.is_file():
        return {"error": f"Model file not found: {MODEL_PATH}"}
    if not isinstance(profile, dict):
        return {"error": "No user profile is selected."}
    if not isinstance(usage, dict):
        return {"error": "No usage data is available for this user."}

    try:
        artifact = load_model()
    except Exception as exc:
        return {"error": f"Could not load model: {exc}"}

    if artifact is None:
        return {"error": "The trained model could not be loaded."}
    if not isinstance(artifact, dict):
        return {
            "error": "Unexpected model artifact format. Expected a dictionary with "
            "'preprocessor', 'model', and 'feature_columns'."
        }

    missing_keys = {"preprocessor", "model", "feature_columns"} - set(artifact.keys())
    if missing_keys:
        return {
            "error": "Model artifact is missing: " + ", ".join(sorted(missing_keys))
        }

    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    feature_columns = list(artifact["feature_columns"])

    features = {
        "age": profile.get("age"),
        "gender": profile.get("gender"),
        "daily_screen_time_hours": usage.get("daily_screen_time_hours", 0.0),
        "social_media_hours": usage.get("social_media_hours", 0.0),
        "gaming_hours": usage.get("gaming_hours", 0.0),
        "work_study_hours": profile.get("work_study_hours", 0.0),
        "sleep_hours": profile.get("sleep_hours", 7.0),
        "notifications_per_day": usage.get("notifications_per_day", 0),
        "app_opens_per_day": usage.get("app_opens_per_day", 0),
        "weekend_screen_time": usage.get(
            "weekend_screen_time", profile.get("weekend_screen_time", 0.0)
        ),
        "stress_level": profile.get("stress_level", "Medium"),
        "academic_work_impact": profile.get("academic_work_impact", "No"),
    }

    missing_features = [
        c for c in feature_columns if c not in features or features[c] is None
    ]
    if missing_features:
        return {"error": "Missing model features: " + ", ".join(missing_features)}

    try:
        input_frame = pd.DataFrame(
            [{c: features[c] for c in feature_columns}], columns=feature_columns
        )
        processed = preprocessor.transform(input_frame)

        if not hasattr(model, "predict_proba"):
            return {"error": "The saved model does not support predict_proba()."}

        probabilities = model.predict_proba(processed)[0]
        classes = list(getattr(model, "classes_", range(len(probabilities))))

        if 1 in classes:
            positive_index = classes.index(1)
        elif len(probabilities) == 2:
            positive_index = 1
        else:
            return {"error": f"Could not identify the risk class in: {classes}"}

        probability = float(probabilities[positive_index])
        return {
            "prediction": int(probability >= 0.50),
            "probability": probability,
            "features": features,
        }
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}


# ============================================================
# TELEGRAM
# ============================================================
def get_telegram_chat_id():
    data = load_json(TELEGRAM_AUTH_PATH)
    return data.get("chat_id") if isinstance(data, dict) else None


def telegram_request(method, data=None):
    if not TELEGRAM_TOKEN:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN is not set."}

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    try:
        response = requests.post(url, data=data or {}, timeout=10)
        result = response.json()
        if not isinstance(result, dict):
            return {"ok": False, "error": "Unexpected Telegram API response."}
        return result
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}
    except ValueError:
        return {"ok": False, "error": "Telegram returned invalid JSON."}


def send_telegram_test():
    chat_id = get_telegram_chat_id()
    if not chat_id:
        return False, "No Telegram account is paired."

    result = telegram_request(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": "🟢 Egate AI test successful.\n\nTelegram alerts are working.",
        },
    )
    if result.get("ok"):
        return True, "Test alert sent successfully."
    return False, result.get(
        "description", result.get("error", "Telegram send failed.")
    )


# ============================================================
# BACKGROUND MONITOR CONTROL
# ============================================================
def monitor_processes():
    """Return processes running background_monitor.py."""
    matches = {}
    target = MONITOR_SCRIPT_NAME.lower()

    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command_line = process.info.get("cmdline") or []
            if any(Path(str(part)).name.lower() == target for part in command_line):
                matches[process.pid] = process
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
            OSError,
        ):
            continue

    return list(matches.values())


def monitor_is_running():
    return bool(monitor_processes())


def start_background_monitor():
    if not MONITOR_PATH.is_file():
        return False, f"Monitor script not found:\n{MONITOR_PATH}"

    if monitor_is_running():
        return True, "Background monitor is already running."

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        with LOG_FILE.open("a", encoding="utf-8") as log_handle:
            subprocess.Popen(
                [sys.executable, str(MONITOR_PATH)],
                cwd=str(SRC_DIR),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )

        return True, "Monitor launch requested. Check the log below if it exits."
    except Exception as exc:
        return False, f"Could not start monitor: {exc}"


def stop_background_monitor():
    processes = monitor_processes()
    if not processes:
        return True, "Background monitor is not running."

    stopped, errors = [], []

    for process in processes:
        try:
            process.terminate()
            stopped.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except (psutil.AccessDenied, OSError) as exc:
            errors.append(f"PID {process.pid}: {exc}")

    if stopped:
        _, alive = psutil.wait_procs(stopped, timeout=5)
        for process in alive:
            try:
                process.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            except (psutil.AccessDenied, OSError) as exc:
                errors.append(f"PID {process.pid}: {exc}")

    if errors:
        return False, "Some processes could not be stopped: " + "; ".join(errors)
    return True, "Background monitor stop request completed."


# ============================================================
# LOAD CURRENT STATE
# ============================================================
profiles = load_profiles()
active_profile_name = get_active_profile_name()

if active_profile_name not in profiles and profiles:
    active_profile_name = next(iter(profiles))
    set_active_profile_name(active_profile_name)
    activate_profile(active_profile_name, profiles[active_profile_name])

profile = profiles.get(active_profile_name) if active_profile_name else None
usage = get_selected_usage(active_profile_name) if active_profile_name else None
chat_id = get_telegram_chat_id()

model_artifact, model_load_error = get_model_artifact()

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">NIP</div>
        <div class="hero-subtitle">AI-assisted digital wellbeing &amp; safety monitor</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Control Center")

    if active_profile_name:
        st.success(f"👤 Active user: {active_profile_name}")
    else:
        st.warning("No active user")

    st.divider()

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    if st.button("▶️ Start Monitor", use_container_width=True):
        ok, message = start_background_monitor()
        st.success(message) if ok else st.error(message)

    if st.button("⏹ Stop Monitor", use_container_width=True):
        ok, message = stop_background_monitor()
        st.success(message) if ok else st.error(message)

    st.divider()
    st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"Data folder: {DATA_DIR}")

# ============================================================
# SYSTEM STATUS
# ============================================================
st.subheader("System Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Monitor", "🟢 ONLINE" if monitor_is_running() else "🔴 OFFLINE")
with col2:
    st.metric("AI Model", "READY" if model_artifact is not None else "MISSING / ERROR")
with col3:
    st.metric("Telegram", "CONNECTED" if chat_id else "NOT PAIRED")
with col4:
    st.metric("Usage Data", "AVAILABLE" if usage else "WAITING")

if model_artifact is None:
    if model_load_error:
        st.warning(f"Could not load the model: {model_load_error}")
    else:
        st.info(f"Model file not found at: {MODEL_PATH}")

# ============================================================
# USER MANAGEMENT
# ============================================================
st.divider()
st.subheader("👤 User Management")
left, right = st.columns([1.2, 1])

with left:
    if profiles:
        profile_names = list(profiles.keys())
        current_index = (
            profile_names.index(active_profile_name)
            if active_profile_name in profile_names
            else 0
        )
        selected_name = st.selectbox("Active user", profile_names, index=current_index)

        if selected_name != active_profile_name:
            active_profile_name = selected_name
            profile = profiles[selected_name]
            set_active_profile_name(selected_name)
            activate_profile(selected_name, profile)
            st.rerun()
    else:
        st.info("No users yet. Create the first profile.")

with right:
    st.caption("Each person gets their own profile and usage record.")
    st.caption("Changing the active user changes which data the model analyzes.")

# ============================================================
# PROFILE EDITOR
# ============================================================
with st.expander("➕ Create / Edit User Profile", expanded=(profile is None)):
    defaults = normalize_profile(profile or {})
    name_value = active_profile_name or ""

    profile_name = st.text_input(
        "Profile name",
        value=name_value,
        placeholder="Example: Alex",
        key="profile_name_input",
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=5,
            max_value=100,
            value=max(5, min(100, defaults["age"])),
            step=1,
        )
        gender_options = ["Male", "Female", "Other"]
        gender = st.selectbox(
            "Gender",
            gender_options,
            index=(
                gender_options.index(defaults["gender"])
                if defaults["gender"] in gender_options
                else 0
            ),
        )
        sleep = st.number_input(
            "Average sleep (hours)",
            min_value=0.0,
            max_value=24.0,
            value=max(0.0, min(24.0, defaults["sleep_hours"])),
            step=0.5,
        )

    with col2:
        work = st.number_input(
            "Work / study (hours/day)",
            min_value=0.0,
            max_value=24.0,
            value=max(0.0, min(24.0, defaults["work_study_hours"])),
            step=0.5,
        )
        stress_options = ["Low", "Medium", "High"]
        stress = st.selectbox(
            "Stress level",
            stress_options,
            index=(
                stress_options.index(defaults["stress_level"])
                if defaults["stress_level"] in stress_options
                else 1
            ),
        )
        impact_options = ["No", "Yes"]
        impact = st.selectbox(
            "Academic/work impact",
            impact_options,
            index=(
                impact_options.index(defaults["academic_work_impact"])
                if defaults["academic_work_impact"] in impact_options
                else 0
            ),
        )

    weekend_profile = st.number_input(
        "Typical weekend screen time (hours)",
        min_value=0.0,
        max_value=24.0,
        value=max(0.0, min(24.0, defaults["weekend_screen_time"])),
        step=0.5,
    )

    if st.button("💾 Save Profile", use_container_width=True):
        clean_name = profile_name.strip()

        if not clean_name:
            st.error("Enter a profile name.")
        else:
            new_profile = {
                "age": int(age),
                "gender": gender,
                "work_study_hours": float(work),
                "sleep_hours": float(sleep),
                "stress_level": stress,
                "academic_work_impact": impact,
                "weekend_screen_time": float(weekend_profile),
            }
            profiles[clean_name] = new_profile
            save_profiles(profiles)
            set_active_profile_name(clean_name)
            activate_profile(clean_name, new_profile)
            st.success(f"✅ {clean_name} is now the active user.")
            st.rerun()

# Refresh state after a profile save/switch.
profiles = load_profiles()
active_profile_name = get_active_profile_name()
profile = profiles.get(active_profile_name) if active_profile_name else None
usage = get_selected_usage(active_profile_name) if active_profile_name else None
chat_id = get_telegram_chat_id()

# ============================================================
# USER SUMMARY
# ============================================================
if profile:
    st.subheader(f"👤 {active_profile_name}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Age", profile["age"])
    with c2:
        st.metric("Gender", profile["gender"])
    with c3:
        st.metric("Sleep", f"{profile['sleep_hours']:.1f} h")
    with c4:
        st.metric("Stress", profile["stress_level"])

# ============================================================
# USAGE DATA
# ============================================================
st.divider()
st.subheader("📊 Usage Data")
st.caption(
    "Usage can currently be entered per user. A compatible collector can update "
    "these values automatically later."
)

with st.expander("✏️ Enter / update this user's usage", expanded=(usage is None)):
    current_usage = usage or {}

    c1, c2, c3 = st.columns(3)

    with c1:
        screen_time = st.number_input(
            "Daily screen time (hours)",
            min_value=0.0,
            max_value=24.0,
            value=max(
                0.0, min(24.0, float(current_usage.get("daily_screen_time_hours", 0.0)))
            ),
            step=0.1,
        )
        social_media = st.number_input(
            "Social media (hours)",
            min_value=0.0,
            max_value=24.0,
            value=max(
                0.0, min(24.0, float(current_usage.get("social_media_hours", 0.0)))
            ),
            step=0.1,
        )

    with c2:
        gaming = st.number_input(
            "Gaming (hours)",
            min_value=0.0,
            max_value=24.0,
            value=max(0.0, min(24.0, float(current_usage.get("gaming_hours", 0.0)))),
            step=0.1,
        )
        notifications = st.number_input(
            "Notifications/day",
            min_value=0,
            max_value=10000,
            value=max(
                0, min(10000, int(current_usage.get("notifications_per_day", 0)))
            ),
            step=1,
        )

    with c3:
        app_opens = st.number_input(
            "App opens/day",
            min_value=0,
            max_value=10000,
            value=max(0, min(10000, int(current_usage.get("app_opens_per_day", 0)))),
            step=1,
        )
        default_weekend = current_usage.get(
            "weekend_screen_time",
            profile.get("weekend_screen_time", 0.0) if profile else 0.0,
        )
        weekend_usage = st.number_input(
            "Weekend screen time (hours)",
            min_value=0.0,
            max_value=24.0,
            value=max(0.0, min(24.0, float(default_weekend))),
            step=0.1,
        )

    if active_profile_name:
        if st.button(
            f"💾 Save usage for {active_profile_name}", use_container_width=True
        ):
            new_usage = {
                "daily_screen_time_hours": float(screen_time),
                "social_media_hours": float(social_media),
                "gaming_hours": float(gaming),
                "notifications_per_day": int(notifications),
                "app_opens_per_day": int(app_opens),
                "weekend_screen_time": float(weekend_usage),
            }
            save_selected_usage(active_profile_name, new_usage)
            st.success(f"✅ Usage saved for {active_profile_name}.")
            st.rerun()
    else:
        st.info("Create a user profile before saving usage data.")

if usage:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Screen time", f"{float(usage.get('daily_screen_time_hours', 0)):.2f} h"
        )
    with c2:
        st.metric("Notifications", usage.get("notifications_per_day", 0))
    with c3:
        st.metric("App opens", usage.get("app_opens_per_day", 0))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Social media", f"{float(usage.get('social_media_hours', 0)):.2f} h")
    with c2:
        st.metric("Gaming", f"{float(usage.get('gaming_hours', 0)):.2f} h")
    with c3:
        st.metric("Weekend", f"{float(usage.get('weekend_screen_time', 0)):.2f} h")
else:
    st.warning("No usage data is available for this user yet.")

# ============================================================
# AI RISK ANALYSIS
# ============================================================
st.divider()
st.subheader("🧠 AI Risk Analysis")

if st.button("🧠 Analyze Current User", type="primary", use_container_width=True):
    if not active_profile_name:
        st.error("Create/select a user first.")
    elif profile is None:
        st.error("The active user's profile is missing.")
    elif usage is None:
        st.error("No usage data exists for this user.")
    else:
        result = run_prediction(profile, usage)

        if "error" in result:
            st.error(result["error"])
        else:
            probability = result["probability"]
            c1, c2 = st.columns(2)

            with c1:
                st.metric("Estimated Risk", f"{probability:.1%}")
                st.progress(max(0.0, min(1.0, probability)))

            with c2:
                if probability >= 0.70:
                    st.error("⚠️ HIGH RISK")
                elif probability >= 0.50:
                    st.warning("🟡 ELEVATED RISK")
                else:
                    st.success("🟢 LOWER RISK")

            if probability >= 0.70:
                st.error(
                    "The model estimates a high-risk usage pattern. "
                    "Review the underlying usage values before taking action."
                )
            elif probability >= 0.50:
                st.warning("The model estimates an elevated-risk usage pattern.")
            else:
                st.success("The model estimates a lower-risk usage pattern.")

            with st.expander("🔎 See values sent to the model"):
                feature_df = pd.DataFrame([result["features"]]).T
                feature_df.columns = ["Value"]
                st.dataframe(feature_df, use_container_width=True)

# ============================================================
# TELEGRAM PAIRING
# ============================================================
st.divider()
st.subheader("📱 Connect Telegram")

chat_id = get_telegram_chat_id()

if chat_id:
    st.success("✅ A Telegram account is paired.")
    st.caption(f"Telegram Chat ID: ••••••••{str(chat_id)[-4:]}")

    if st.button("📨 Send Test Alert", use_container_width=True):
        ok, message = send_telegram_test()
        st.success(message) if ok else st.error(message)
else:
    st.warning("No family Telegram account is paired.")

st.markdown("""
    **Pair your Telegram account**

    1. Click **Start Monitor** in the sidebar (it must be running).
    2. Open your Egate AI bot in Telegram and send `/start`.
    3. Enter the six-digit code below.
    4. Click **Verify and Pair Telegram**.
    """)

with st.form("telegram_pairing_form"):
    registration_code = st.text_input(
        "Six-digit registration code",
        placeholder="Enter the code from Telegram",
        max_chars=6,
    )
    submitted = st.form_submit_button(
        "🔐 Verify and Pair Telegram", use_container_width=True
    )

    if submitted:
        code = registration_code.strip()

        if len(code) != 6 or not code.isdigit():
            st.error("Enter exactly six digits.")

        elif not monitor_is_running():
            st.error(
                "The background monitor is not running. Start it, send /start "
                "to your bot, and submit the new code."
            )

        else:
            try:
                # Remove an old result so it can't be mistaken for this one.
                if TELEGRAM_PAIRING_STATUS_PATH.exists():
                    TELEGRAM_PAIRING_STATUS_PATH.unlink()

                save_json(TELEGRAM_CODE_SUBMIT_PATH, {"code": code})

                # Wait (up to ~10 s) for the monitor's answer.
                result_status = None
                with st.spinner("Checking the code with the monitor..."):
                    for _ in range(20):
                        time.sleep(0.5)
                        result_status = load_json(TELEGRAM_PAIRING_STATUS_PATH)
                        if isinstance(result_status, dict):
                            break

                if not isinstance(result_status, dict):
                    st.error(
                        "The monitor did not answer. Check the monitor log below "
                        "(it may have crashed or still be loading)."
                    )
                elif result_status.get("status") == "success":
                    st.rerun()
                # Errors are shown by the status block below.

            except OSError as exc:
                st.error(f"Could not submit the code: {exc}")

pairing_status = load_json(TELEGRAM_PAIRING_STATUS_PATH)

if isinstance(pairing_status, dict):
    status = pairing_status.get("status")
    message = pairing_status.get("message", "")

    if status == "success":
        st.success(f"✅ {message}")
    elif status == "error":
        st.error(f"❌ {message}")

if not TELEGRAM_TOKEN:
    st.warning(
        "TELEGRAM_BOT_TOKEN is not available to the dashboard. Put it in "
        f"`{SRC_DIR / '.env'}` as: TELEGRAM_BOT_TOKEN=your_token"
    )

# ============================================================
# SAFETY MONITOR AND LOG
# ============================================================
st.divider()
st.subheader("🛡️ Safety Monitor")

if monitor_is_running():
    st.success("🟢 background_monitor.py is running.")
else:
    st.warning("🟡 background_monitor.py is not running.")

st.caption(
    "NSFW detection and interventions run in background_monitor.py, separately "
    "from the dashboard's risk prediction."
)

with st.expander("📋 Recent monitor log"):
    if LOG_FILE.is_file():
        try:
            with LOG_FILE.open("r", encoding="utf-8", errors="replace") as file:
                lines = file.readlines()
            st.code("".join(lines[-40:]) or "Log file is empty.", language="text")
        except OSError as exc:
            st.error(f"Could not read log: {exc}")
    else:
        st.info(f"No monitor log exists yet. Expected location: {LOG_FILE}")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Model output is a statistical estimate based on the training data; "
    "it is not a diagnosis."
)
