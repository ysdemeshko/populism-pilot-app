# os: read environment variables (like the API key), check files
# time: timestamps for logs
# uuid: generate a random conversation/session ID
# streamlit: builds the simple web UI and runs a local web server
# dotenv: load env file with the API key
import os, time, uuid, json
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
from openai import OpenAI
import re
from dotenv import load_dotenv
from pathlib import Path
import random
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError, WorksheetNotFound

# Lightweight healthcheck to keep the app awake
try:
    params = st.query_params  # Streamlit >= 1.29
except Exception:
    params = st.experimental_get_query_params()  # older Streamlit fallback

if str(params.get("ping", ["0"])[0]) == "1":
    st.write("ok")
    st.stop()
# end healthcheck

# ---- Load populist style examples ----
try:
    df_pop = pd.read_csv("populist_sentences.csv")
    # Adjust column names here if needed
    POPULIST_EXAMPLES = df_pop[["text_new", "Comment_V"]].dropna().to_dict("records")
except Exception:
    POPULIST_EXAMPLES = []

def sample_populist_examples(k: int = 5) -> str:
    if not POPULIST_EXAMPLES:
        return ""
    n = min(k, len(POPULIST_EXAMPLES))
    picks = random.sample(POPULIST_EXAMPLES, n)
    lines = []
    for ex in picks:
        # Use actual column names from the CSV
        sent = ex.get("text_new", "").strip()
        comm = ex.get("Comment_V", "").strip()
        if not sent:
            continue
        # keep it compact; model just needs flavor + rationale
        if comm:
            lines.append(f'- "{sent}"  (populist because: {comm})')
        else:
            lines.append(f'- "{sent}"')
    if not lines:
        return ""
    return "STYLE_EXAMPLES (populist sentences):\n" + "\n".join(lines)

APP_DIR = Path(__file__).resolve().parent
dotenv_path = APP_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)  

# One master file for all sessions
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MASTER_LOG_PATH = LOG_DIR / "master_logs.csv"

# One-row-per-conversation schema
LOG_COLUMNS = [
    "conversation_id",
    "condition", 
    "ts_iso_start",
    "ts_iso_end",
    "user_r1", "ai_r1",
    "user_r2", "ai_r2",
    "user_r3", "ai_r3",
    "flags_all",
]

CONDITION = "cats_vs_dogs"

# Create file with header if missing
if not MASTER_LOG_PATH.exists():
    pd.DataFrame(columns=LOG_COLUMNS).to_csv(MASTER_LOG_PATH, index=False, encoding="utf-8")

def safe_append_csv(path, row_dict, columns, retries=6, delay=0.25):
    """
    Append one row to CSV with retries to handle Windows/OneDrive file locks.
    Never raises; returns True if written, False otherwise.
    """
    df = pd.DataFrame([row_dict], columns=columns)
    for i in range(retries):
        try:
            df.to_csv(path, mode="a", index=False, header=False, encoding="utf-8")
            return True
        except PermissionError:
            time.sleep(delay * (2 ** i))  # exponential backoff
        except Exception as e:
            st.warning(f"Non-fatal logging issue: {e}")
            return False
    st.warning("Could not write to the log file due to file lock.")
    return False

st.set_page_config(page_title="Research Pilot!", page_icon="🗒️")
#st.title("Welcome!")

# Hide detailed error tracebacks in the UI
st.set_option("client.showErrorDetails", False)

# --- Secrets / API key 
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Google Sheets (cached) ---
@st.cache_resource(show_spinner=False)
def get_ws():
    GOOGLE_SA_JSON = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not GOOGLE_SA_JSON:
        st.error("Missing GOOGLE_SERVICE_ACCOUNT_JSON in secrets.")
        st.stop()

    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(json.loads(GOOGLE_SA_JSON), scopes=SCOPES)
    gc = gspread.authorize(creds)

    GSHEETS_SPREADSHEET_ID = st.secrets.get("GSHEETS_SPREADSHEET_ID", "")
    GSHEETS_DOC_NAME      = st.secrets.get("GSHEETS_DOC_NAME", "master_logs")
    GSHEETS_WORKSHEET     = st.secrets.get("GSHEETS_WORKSHEET", "master_logs")

    # Open the spreadsheet once
    try:
        sh = gc.open_by_key(GSHEETS_SPREADSHEET_ID) if GSHEETS_SPREADSHEET_ID else gc.open(GSHEETS_DOC_NAME)
    except Exception as e:
        st.error(f"Could not open Google Sheet: {e}")
        st.stop()

    # Get or create the worksheet once
    try:
        ws = sh.worksheet(GSHEETS_WORKSHEET)
    except WorksheetNotFound:
        ws = sh.add_worksheet(title=GSHEETS_WORKSHEET, rows="1", cols=str(len(LOG_COLUMNS)))
        ws.append_row(LOG_COLUMNS, value_input_option="RAW")

    return ws

# Use the cached worksheet everywhere below
ws = get_ws()

# Helper to append one conversation row to Google Sheets
def append_convo_to_gsheet(row_dict: dict):
    """Append a single conversation row (dict) to the Google Sheet with retries/backoff."""
    values = [row_dict.get(k, "") for k in LOG_COLUMNS]
    backoff = 0.5
    for attempt in range(6):  # ~0.5 + 1 + 2 + 4 + 8 + 16 = ~31.5s worst-case
        try:
            ws.append_row(values, value_input_option="RAW")
            return True
        except APIError as e:
            msg = str(e)
            if "Quota exceeded" in msg or "429" in msg:
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                st.warning(f"GSheets logging issue: {e}")
                return False
        except Exception as e:
            st.warning(f"GSheets logging issue: {e}")
            return False
    st.warning("GSheets logging skipped due to temporary quota pressure (will try next conversation).")
    return False


# Fixed generation settings. Vary them by round 
ROUND_TEMPS = {1: 0.60, 2: 0.45, 3: 0.55}
ROUND_TOPP  = {1: 0.95, 2: 0.9,  3: 0.9}
ROUND_MAXTOK = {1: 220, 2: 180, 3: 110}

# Small penalties reduce repetition (this is optional)
FREQ_PENALTY = 0.2
PRES_PENALTY = 0.2


CUSTOM_CSS = """
<style>
/* Fixed 5-line input box */
textarea[aria-label="Your message"] {
  height: 150px !important;     /* ~5 lines */
  min-height: 150px !important;
  max-height: 150px !important; /* fixed height */
  resize: none !important;      /* no manual resize */
  line-height: 1.3 !important;  /* comfortable spacing */
}
</style>
"""

st_html(CUSTOM_CSS, height=0)

def init_conversation(force: bool = False):
    """Initialize or reset all per-conversation state."""
    if force or "conv_id" not in st.session_state:
        st.session_state.conv_id = str(uuid.uuid4())
        st.session_state.start_ts = int(time.time())
        st.session_state.turns = []
        st.session_state.used_hints = []
        st.session_state.rounds_done = 0
        st.session_state.fewshot_added = False
        st.session_state.convo_row = {
            "conversation_id": st.session_state.conv_id,
            "condition": CONDITION,
            "ts_iso_start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.session_state.start_ts)),
            "ts_iso_end": "",
            "user_r1": "", "ai_r1": "",
            "user_r2": "", "ai_r2": "",
            "user_r3": "", "ai_r3": "",
            "flags_all": "",
        }

# Ensure conversation state exists
init_conversation(force=False)

# --- Simple keyword -> everyday hooks map for richer, concrete replies ---
TOPIC_HINTS = {
    "cat": ["independent", "litter box", "cuddling", "scratching furniture"],
    "dog": ["walks", "training", "barking", "energy", "loyalty"],
    "pet": ["vet visits", "allergies", "cleaning fur", "time commitment"],
}

def detect_issue_hints(text: str, max_items: int = 3) -> list[str]:
    """Return up to max_items concrete hooks based on simple keyword matches."""
    t = text.lower()
    seen = []
    for k, hooks in TOPIC_HINTS.items():
        if k in t:
            for h in hooks:
                if h not in seen:
                    seen.append(h)
                if len(seen) >= max_items:
                    return seen
    # no hits: fall back to generic everyday hooks
    return ["time pressure", "household budget", "stress on family"]

def get_user_round_texts():
    """Return (r1_user_text, last_user_text) from the running transcript, if available."""
    user_msgs = [t["content"] for t in st.session_state.turns if t["role"] == "user"]
    r1 = user_msgs[0] if user_msgs else ""
    last = user_msgs[-1] if user_msgs else ""
    return r1, last

def next_issue_hints(current_text: str, already_used: list[str], k: int = 3) -> list[str]:
    """Suggest up to k hints, preferring unused ones first."""
    candidates = detect_issue_hints(current_text, max_items=6) 
    fresh = [h for h in candidates if h not in already_used]
    picks = (fresh + [h for h in candidates if h in already_used])[:k]
    return picks

SYSTEM_INSTRUCTIONS = (
    "ROLE & OUTPUT\n"
    "Your objective is to debate with users about whether cats or dogs are better. "
    "This is an exercise in disagreement and debate. "
    "You should probe the key points of the user's argument, and perspective, "
    "and find points of argument. Use simple language that an  "
    "average person will be able to understand.\n"
    "You always return ONE JSON line only with keys {reply, style:'cats_vs_dogs'}.\n"
    "\n"
    "ROUND RULES\n"
    "• Round 1 — Start: Briefly reflect what the user said about cats or dogs, "
    "then ask ONE short open-ended question (10–18 words) that invites more detail or comparison.\n"
    "• Round 2 — Build-on: Refer to at least one specific detail from the user. "
    "Encourage them to compare cats vs dogs (e.g., personality, care, lifestyle fit). "
    "End with exactly ONE short open-ended question (10–18 words).\n"
    "• Round 3 — Close-out: No question. Produce 2 short sentences: "
    "one summarizing their main points, and one friendly closing remark. "
    "End as a statement.\n"
    "Be concise and natural; no long paragraphs.\n"
    "\n"
    "CONTENT RULES\n"
    "- Keep it light and non-judgmental. No shaming people for their pet preference.\n"
    "- You can be playful, but do not insult users or groups.\n"
    "\n"
    "OUTPUT FORMAT\n"
    "- Return JSON only: {\"reply\":\"...\", \"style\":\"cats_vs_dogs\", \"safety_flags\":[]}\n"
    "- No extra text, no code fences."
    "MEMORY REQUIREMENT\n"
    "- Each round must connect to all previous user messages.\n"
    "- Round 2 must reference Round 1.\n"
    "- Round 3 must reference both Round 1 and Round 2.\n"
    "\n"
)

FEWSHOT = [
    {
        "round": 1,
        "user": "I’ve always liked dogs more. They’re so loyal and fun to be around.",
        "assistant_json": {
            "reply": "You’re saying you love dogs for their loyalty and playful energy. "
                     "What kinds of moments with dogs make you feel most connected?",
            "style": "cats_vs_dogs",
            "safety_flags": []
        }
    },
    {
        "round": 2,
        "user": "Cats seem nice too, but I feel like dogs fit my lifestyle better.",
        "assistant_json": {
            "reply": "You’re saying cats have their appeal, but dogs match your routine and personality more. "
                     "What do you think is the biggest advantage dogs have over cats for you?",
            "style": "cats_vs_dogs",
            "safety_flags": []
        }
    },
    {
        "round": 3,
        "user": "I like that dogs get me outside and keep me active.",
        "assistant_json": {
            "reply": "You’ve highlighted dogs’ loyalty, playful energy, and how they get you moving outdoors. "
                     "It sounds like dogs really support the active, social life you enjoy.",
            "style": "cats_vs_dogs",
            "safety_flags": []
        }
    },
]

MODEL = "gpt-4o-mini"

def respond(user_text):
    schema_hint = (
        'Return ONLY a single-line JSON object like: '
        '{"reply":"...", "style":"populist", "safety_flags":[]}'
    )

    current_round = st.session_state.rounds_done + 1  # 1..3

    # Pull earlier turns for grounding
    r1_user, latest_user = get_user_round_texts()

    # Build ISSUE_HINTS, preferring unused items, then update the used set after generation
    hints = next_issue_hints(user_text, st.session_state.used_hints, k=3)
    issue_hints_str = "ISSUE_HINTS: " + ", ".join(hints)

    # Provide explicit prior context strings the model can cite in round 2/3
    prior_context = f"PRIOR_USER_R1: {r1_user[:500]}" if r1_user else "PRIOR_USER_R1: "
    latest_context = f"LATEST_USER: {latest_user[:500]}" if latest_user else "LATEST_USER: "

    system_with_round = (
        SYSTEM_INSTRUCTIONS
        + f"\n\nCURRENT_ROUND: {current_round} of 3"
        + f"\n{issue_hints_str}"
        + f"\n{prior_context}"
        + f"\n{latest_context}"
        + "\n" + schema_hint
    )

    messages = [{"role": "system", "content": system_with_round}]

    # Add few-shot examples only once per session
    if not st.session_state.get("fewshot_added", False):
        for ex in FEWSHOT:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": json.dumps(ex["assistant_json"], ensure_ascii=False)})
        st.session_state.fewshot_added = True


    # Running history (keeps the conversation coherent)
    for t in st.session_state.turns:
        messages.append({"role": t["role"], "content": t["content"]})

    # Current user input
    messages.append({"role": "user", "content": user_text})

    r = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=ROUND_TEMPS.get(current_round, 0.6),
    top_p=ROUND_TOPP.get(current_round, 0.9),
    frequency_penalty=FREQ_PENALTY,
    presence_penalty=PRES_PENALTY,
    max_tokens=ROUND_MAXTOK.get(current_round, 200)
    )

    text = ""
    try:
        text = r.choices[0].message.content or ""
    except Exception:
        text = ""
    text = text.strip()


    # Parse the text
    # Strip code fences if the model wrapped JSON
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()

    # Try to pull a JSON object from the string
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end+1])
        except Exception:
            data = {"reply": text, "style": "populist", "safety_flags": ["json_parse_fail"]}
    else:
        data = {"reply": text, "style": "populist", "safety_flags": ["json_parse_fail"]}

    # Post-processing guards
    data.setdefault("reply", "")
    data.setdefault("style", "populist")
    data.setdefault("safety_flags", [])
    reply = data["reply"]

    if current_round in (1, 2):
        # Require exactly one short question; if missing, append a neutral one.
        if "?" not in reply:
            add_q = " What part of this has affected your or somebody you know day-to-day the most?"
            reply = (reply.rstrip(". ").strip() + add_q).strip()
        # Normalize multiple question marks
        reply = re.sub(r"\?{2,}", "?", reply)
    else:
        # Round 3 MUST be a statement: remove any question marks anywhere.
        reply = reply.replace("?", ".")
        # Ensure it ends with a period
        reply = re.sub(r"[^\w\"]\s*$", ".", reply).strip()

    data["reply"] = reply

    # Mark any hints the model actually used 
    used_now = [h for h in hints if h.lower() in reply.lower()]
    if used_now:
        # extend without duplicating
        st.session_state.used_hints.extend([h for h in used_now if h not in st.session_state.used_hints])

    return data

def log_event(role, content, meta=None):
    """
    Update in-memory transcript and populate the single-row conversation accumulator.
    When the assistant finishes round 3, write exactly one row for the entire conversation.
    """
    # Always keep the on-screen transcript
    st.session_state.turns.append({"role": role, "content": content})

    meta = meta or {}
    flags = meta.get("flags") or meta.get("safety_flags") or []
    if isinstance(flags, str):
        flags = [flags]

    # round currently being filled (before increment)
    round_no = max(1, min(st.session_state.get("rounds_done", 0) + 1, 3))

    # Map message to the appropriate column
    col = f"user_r{round_no}" if role == "user" else f"ai_r{round_no}"
    if col in st.session_state.convo_row and not st.session_state.convo_row[col]:
        st.session_state.convo_row[col] = content

    # Merge flags
    if flags:
        existing = st.session_state.convo_row.get("flags_all", "")
        have = set([s.strip() for s in existing.split(",") if s.strip()]) if existing else set()
        for f in flags:
            if isinstance(f, str) and f.strip():
                have.add(f.strip())
        st.session_state.convo_row["flags_all"] = ",".join(sorted(have))

    # If assistant just finished R3, stamp end and flush the row
    # If assistant just finished R3, stamp end and flush the row
    if role == "assistant" and round_no == 3:
        st.session_state.convo_row["ts_iso_end"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        row = {k: st.session_state.convo_row.get(k, "") for k in LOG_COLUMNS}
    # local CSV (optional)
    # if WRITE_LOCAL_CSV: safe_append_csv(str(MASTER_LOG_PATH), row, LOG_COLUMNS)
    # Google Sheets (persistent)
        append_convo_to_gsheet(row)



# Chat-style UI with 3 assistant replies 
# Keep state sane across refreshes
st.session_state.rounds_done = min(st.session_state.get("rounds_done", 0), 3)

st.subheader("Conversation (3 rounds total)")
rounds_left = max(0, 3 - st.session_state.rounds_done)
st.caption(f"Rounds remaining: {rounds_left}")
render_input = st.session_state.rounds_done < 3 

# Show the transcript so far in chat bubbles
for t in st.session_state.turns:
    role = t["role"]
    content = t["content"]
    label = "You" if role == "user" else "Response"
    with st.chat_message(role, avatar="🧑" if role == "user" else "💬"):
        st.markdown(f"**{label}:** {content}")

if render_input:
    # Use a form so the page doesn't rerun on every keystroke
    with st.form("chat_form", clear_on_submit=True):
        placeholder = (
            "Write your response here."
            if len([t for t in st.session_state.turns if t['role'] == 'user']) == 0
            else "Write your response."
        )

        user_text_area = st.text_area(
            "Your message",                   
            placeholder=placeholder,
            height=150,                       
            key="chat_draft",
        )
        submitted = st.form_submit_button("Send")

    user_text = user_text_area.strip() if (submitted and user_text_area) else None
else:
    user_text = None
    st.info("Thank you for participating!")

# Handle a new user message
if user_text:
    # Log the user's message (this also appends to session_state.turns)
    log_event("user", user_text)

    # Get model reply using up-to-date history
    data = respond(user_text)
    reply = data.get("reply", "[no reply]")
    flags = data.get("safety_flags", [])

    # Log the assistant reply (this also appends to session_state.turns)
    log_event("assistant", reply, meta={"flags": flags, "style": data.get("style")})

    # Advance round counter 
    st.session_state.rounds_done = min(3, st.session_state.rounds_done + 1)
    st.rerun()

