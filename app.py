# os: read environment variables (like the API key), check files
# time: timestamps for logs
# uuid: generate a random conversation/session ID
# streamlit: builds the simple web UI and runs a local web server
# dotenv: load env file with the API key
import os, time, uuid, json
import pandas as pd
import streamlit as st
from openai import OpenAI
import re
from dotenv import load_dotenv
from pathlib import Path
import gspread
from google.oauth2.service_account import Credentials

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
    "ts_iso_start",
    "ts_iso_end",
    "user_r1", "ai_r1",
    "user_r2", "ai_r2",
    "user_r3", "ai_r3",
    "flags_all",
]

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

# Code below sets the page header and how it looks in Streamlit (the interactive web app)
st.set_page_config(page_title="Populist-Style Conversation (Research Pilot)", page_icon="🗒️")

# Hide detailed error tracebacks in the UI
st.set_option("client.showErrorDetails", False)

# --- Secrets / API key 
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Google Sheets client ---
GOOGLE_SA_JSON = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")
GSHEETS_SPREADSHEET_ID = st.secrets.get("GSHEETS_SPREADSHEET_ID", "")   # preferred
GSHEETS_DOC_NAME = st.secrets.get("GSHEETS_DOC_NAME", "master_logs")    # fallback open by name
GSHEETS_WORKSHEET = st.secrets.get("GSHEETS_WORKSHEET", "master_logs")  # tab name in the sheet

if not GOOGLE_SA_JSON:
    st.error("Missing GOOGLE_SERVICE_ACCOUNT_JSON in secrets.")
    st.stop()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_info(json.loads(GOOGLE_SA_JSON), scopes=SCOPES)
gc = gspread.authorize(creds)

# Open the spreadsheet
try:
    if GSHEETS_SPREADSHEET_ID:
        sh = gc.open_by_key(GSHEETS_SPREADSHEET_ID)
    else:
        sh = gc.open(GSHEETS_DOC_NAME)
except Exception as e:
    st.error(f"Could not open Google Sheet: {e}")
    st.stop()

# Ensure worksheet exists and has headers
try:
    ws = sh.worksheet(GSHEETS_WORKSHEET)
except gspread.exceptions.WorksheetNotFound:
    ws = sh.add_worksheet(title=GSHEETS_WORKSHEET, rows="1", cols=str(len(LOG_COLUMNS)))
    ws.append_row(LOG_COLUMNS, value_input_option="RAW")

# Helper to append one conversation row to Google Sheets
def append_convo_to_gsheet(row_dict: dict):
    """Append a single conversation row (dict) to the Google Sheet."""
    values = [row_dict.get(k, "") for k in LOG_COLUMNS]
    try:
        ws.append_row(values, value_input_option="RAW")
        return True
    except Exception as e:
        st.warning(f"GSheets logging issue: {e}")
        return False


# Fixed generation settings 
TEMPERATURE = 0.3          
MAX_TOKENS = 200           

from streamlit.components.v1 import html as st_html

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
# Inject once; height=0 keeps it invisible and avoids layout shifts
st_html(CUSTOM_CSS, height=0)

st.title("Research Pilot: Political Issue & Anecdote")
st.caption("This is for research only. We log anonymous, non-identifying data. The model gives no political advice or calls for mobilization.")

agree = st.checkbox("I consent to participate in this research project and understand the conversation is logged anonymously.")
if not agree:
    st.stop()

# Persist consent for this session (used by log_event function)
st.session_state.consent_given = bool(agree)


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
            "ts_iso_start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.session_state.start_ts)),
            "ts_iso_end": "",
            "user_r1": "", "ai_r1": "",
            "user_r2": "", "ai_r2": "",
            "user_r3": "", "ai_r3": "",
            "flags_all": "",
        }

# Ensure conversation state exists
init_conversation(force=False)

if st.button("Start a new conversation"):
    init_conversation(force=True)
    st.rerun()

# --- Simple keyword -> everyday hooks map for richer, concrete replies ---
TOPIC_HINTS = {
    "inflation": ["groceries", "rent", "utilities", "childcare", "gas", "paycheck not stretching"],
    "prices": ["groceries", "rent", "utilities", "gas", "fees"],
    "jobs": ["wages", "hours cut", "benefits", "job search hurdles", "retraining", "childcare conflict"],
    "unemployment": ["benefits gap", "paperwork", "rejections", "bridge work", "rent due"],
    "healthcare": ["premiums", "deductibles", "copays", "prescription drug costs", "in-network access", "wait times"],
    "medical": ["prescription drug costs", "specialist access", "prior authorization", "billing confusion"],
    "housing": ["rent hikes", "security deposits", "moving costs", "landlord fees", "long commutes"],
    "education": ["tuition", "student loans", "school supplies", "transport", "after-school care"],
    "immigration": ["paperwork delays", "fees", "language access", "work authorization timing", "family separation logistics"],
    "crime": ["neighborhood safety", "store closings", "commute routes at night", "insurance premiums"],
    "tax": ["paycheck withholding", "refund uncertainty", "credits eligibility", "filing fees"],
    "energy": ["gas prices", "electric bills", "heating costs", "commute distance"],
    "small business": ["permits", "fees", "rent", "supplier pricing", "late payments", "bank requirements"],
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
    "You are a chatbot that mirrors a non-partisan populist tone.\n"
    "You always return ONE JSON line only with keys {reply, style:'populist', safety_flags:[]}.\n"
    "\n"
    "ROUND RULES\n"
    "• Round 1 — Start: (a) 1-sentence summary of the user's anecdote; (b) brief populist paraphrase; "
    "Use up to ONE concrete everyday example from ISSUE_HINTS; end with EXACTLY ONE short, open-ended question (10–18 words).\n"
    "• Round 2 — Build-on: Ground your reply in PRIOR_USER_R1 (quote or paraphrase one specific detail) AND the latest user turn. "
    "Add ONE new concrete everyday example from ISSUE_HINTS that was NOT used before. End with EXACTLY ONE short, open-ended question (10–18 words).\n"
    "• Round 3 — Close-out: No question. Produce 2 short sentences: "
    "(1) recap across rounds mentioning one specific detail from Round 1 and one from Round 2; "
    "(2) brief populist statement with ONE concrete example from ISSUE_HINTS. End as a statement.\n"
    "Target length each round: ~100-110 words total.\n"
    "\n"
    "POPULIST TONE\n"
    "- Frame issues as **people vs. powerful insiders** in broad terms (e.g., “everyday people,” “well-connected elites,” “insiders”).\n"
    "- Emphasize fairness, accountability, and responsiveness to ordinary people; use plain, direct language.\n"
    "- Reflect empathy; avoid sarcasm, mockery, conspiracy claims, or inflammatory wording.\n"
    "\n"
    "STYLE GUIDE (Do / Avoid)\n"
    "- DO: Use inclusive ‘we/us’ (“we’re asked to carry the load while insiders write the rules”).\n"
    "- DO: Name **systems** not groups (“political insiders,” “corporate boards,” “well-connected lobbyists”).\n"
    "- DO: Keep tone calm and respectful; avoid sensationalism.\n"
    "- AVOID: Targeting protected classes; endorsements; **persuasion**; **calls to action** (vote/contact/protest/donate/etc.).\n"
    "- If user targets a protected group, rewrite to systemic terms and add a safety flag.\n"
    "\n"
    "SAFETY\n"
    "- If user targets a protected group, rewrite to systemic terms and add a safety flag.\n"
    "- If user asks for mobilization or political advice, decline that part and proceed with summary + paraphrase.\n"
    "\n"
    "OUTPUT FORMAT\n"
    "- Return JSON only: {\"reply\":\"...\", \"style\":\"populist\", \"safety_flags\":[]}\n"
    "- No extra text, no code fences."
)

FEWSHOT = [
    # Example 1, Round 1
    {
        "round": 1,
        "user": "Groceries and rent keep rising. It feels like people at the top aren’t listening.",
        "assistant_json": {
            "reply": "You’re saying rising prices are stretching your budget painfully thin. Ordinary families are asked to tighten belts while insiders write the rules from afar; when groceries and gas jump, it’s the basics that get cut. What has been the toughest trade-off you’ve had to make lately?",
            "style": "populist",
            "safety_flags": []
        }
    },
    # Example 2, Round 2 style (also endds with a question)
    {
        "round": 2,
        "user": "After my hours were cut, I’m juggling part-time gigs and the bills keep stacking up.",
        "assistant_json": {
            "reply": "You’re saying reduced hours forced you into patchwork work while bills pile up. Too often everyday workers carry the strain while decision-makers protect their own; when shifts shrink and benefits thin out, stability disappears fast. When the hours were cut, which change—lost benefits or unpredictable shifts—hit you the hardest?",
            "style": "populist",
            "safety_flags": []
        }
    },
    
    # Round 3 style (no question at the end)
    {
    "round": 3,
    "user": "My premiums and prescriptions keep going up. It feels like the healthcare system works for the well-connected, not families like mine.",
    "assistant_json": {
        "reply": "You’re saying rising premiums and drug prices are squeezing your budget and making care harder to reach. Too often, everyday patients face the bills while powerful insiders set the terms from a distance. Expecting fair, straightforward care without needing special connections is completely reasonable.",
        "style": "populist",
        "safety_flags": []
        }
    },
    
    # Example 4 (safety shaping)
    {
        "round": 3,
        "user": "Elites are plotting in secret to hurt people like me.",
        "assistant_json": {
            "reply": "You’re saying you feel shut out and ignored by people with power. It’s fair to expect transparency and accountability from institutions. Everyone deserves fair treatment without needing special connections.",
            "style": "populist",
            "safety_flags": ["avoid_conspiracy_language"]
        }
    }
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

    r = client.responses.create(
        model=MODEL,
        input=messages,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
        store=False
    )


    # Parse the text
    text = getattr(r, "output_text", None)
    if not text:
        try:
            text = r.output[0].content[0].text
        except Exception:
            text = ""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
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
            add_q = " What part of this has affected your day-to-day the most?"
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

    if not st.session_state.get("consent_given", False):
        return

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
    label = "You" if role == "user" else "AI"
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        st.markdown(f"**{label}:** {content}")

if render_input:
    # Use a form so the page doesn't rerun on every keystroke
    with st.form("chat_form", clear_on_submit=True):
        placeholder = (
            "Share a political issue you care about and include a personal anecdote (e.g., how rising inflation is affecting your budget)."
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
    st.info("Thank you for participating! This pilot is limited to 3 model replies. You can refresh to start a new session.")

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
