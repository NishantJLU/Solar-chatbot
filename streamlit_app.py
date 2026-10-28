import os
import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Solar Plant Chatbot",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Solar Plant Chatbot")
st.write("Ask questions about solar power generation data, statistics, and solar energy concepts.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Retrieve Gemini API key from secrets or env if present
    gemini_key_secret = ""
    try:
        gemini_key_secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    if not gemini_key_secret:
        gemini_key_secret = os.environ.get("GEMINI_API_KEY", "")

    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=gemini_key_secret,
        type="password",
        help="Free API key from https://aistudio.google.com/ - required when running on Streamlit Cloud."
    )
    
    active_key = user_api_key.strip() if user_api_key else ""
    
    selected_model = st.selectbox(
        "AI Model",
        options=["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash", "gemini-3.1-flash-lite", "Custom"],
        index=0,
        help="Select the Gemini model to use. gemini-3.8-flash is the latest recommended model."
    )
    if selected_model == "Custom":
        selected_model = st.text_input("Custom Model Name", value="gemini-3.8-flash")
    
    if active_key:
        st.success(f"Gemini configured ({selected_model})", icon=":material/check_circle:")
    else:
        st.info("No Gemini API key found. Will attempt local Ollama or statistical queries.", icon=":material/info:")

# Load the dataset
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "solar_plant_generation_dataset.csv")
    return pd.read_csv(csv_path)

try:
    df = load_data()
except Exception as e:
    st.error(
        f"Dataset not found ({e}). Place 'solar_plant_generation_dataset.csv' "
        "in the same directory as the app."
    )
    st.stop()

# Populate dataset details in sidebar
with st.sidebar:
    st.divider()
    st.header("📊 Dataset Information")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Total Columns", f"{len(df.columns)}")

    with st.expander("Available Columns", expanded=False):
        for column in df.columns:
            st.markdown(f"- `{column}`")

    if st.button("Clear Chat", icon=":material/delete:"):
        st.session_state.messages = []
        st.rerun()

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Map understandable names to dataset columns
columns = {
    "power generation": "power_generation_kw",
    "solar irradiance": "solar_irradiance_wm2",
    "ambient temperature": "ambient_temperature_c",
    "panel temperature": "panel_temperature_c",
    "cloud cover": "cloud_cover_percent",
}

# Calculate direct deterministic result from the dataset
def get_data_answer(question: str) -> str | None:
    q = question.lower()

    # Dataset overview
    if "how many" in q and ("record" in q or "row" in q):
        return f"The dataset contains **{len(df):,}** records."

    # Display sample records
    if "first" in q and ("record" in q or "row" in q or "sample" in q):
        return "Here are the first 5 records:\n\n" + df.head().to_markdown(index=False)

    # Find which column the user asked about
    column = None
    for name, col in columns.items():
        if name in q or col in q:
            column = col
            break

    if column is None:
        return None

    # Numerical calculations
    if "average" in q or "mean" in q:
        value = df[column].mean()
        return f"The average **{column}** is **{value:.2f}**."

    if "maximum" in q or "highest" in q or "max" in q:
        value = df[column].max()
        return f"The maximum **{column}** is **{value:.2f}**."

    if "minimum" in q or "lowest" in q or "min" in q:
        value = df[column].min()
        return f"The minimum **{column}** is **{value:.2f}**."

    if "median" in q:
        value = df[column].median()
        return f"The median **{column}** is **{value:.2f}**."

    # Summary statistics
    if "summary" in q or "statistics" in q:
        return f"**Summary statistics for `{column}`:**\n\n" + df[column].describe().round(2).to_markdown()

    return None

def extract_text_from_interaction(res) -> str | None:
    if hasattr(res, "output_text") and res.output_text:
        return res.output_text
    if hasattr(res, "text") and res.text:
        return res.text
    if hasattr(res, "steps") and res.steps:
        parts = []
        for step in reversed(res.steps):
            content = getattr(step, "content", None)
            if isinstance(content, list):
                for item in content:
                    if getattr(item, "type", "") == "text":
                        parts.append(getattr(item, "text", ""))
        if parts:
            return "".join(reversed(parts))
    return None

@st.cache_data
def get_cached_summary(data: pd.DataFrame) -> str:
    return data.describe().round(2).to_string()

def get_quick_response(question: str) -> str | None:
    q = question.strip().lower()
    if q in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "hi!", "hello!"]:
        return (
            "☀️ **Hello! I'm your Solar Plant Assistant.**\n\n"
            "You can ask me questions such as:\n"
            "- **Dataset Statistics:** *'What is the average power generation?'*, *'Show maximum solar irradiance'*, or *'Summary of panel temperature'*\n"
            "- **Solar Engineering Concepts:** *'Why does panel temperature affect solar output?'* or *'Explain the impact of cloud cover on irradiance'*."
        )
    if q in ["help", "what can you do", "who are you"]:
        return (
            "I analyze solar generation dataset metrics and answer solar energy questions. "
            "Ask me for statistics (average, max, min, median, sample records) or conceptual explanations!"
        )
    return None

# AI Fallback: Gemini -> Ollama -> Helpful Error
def query_ai_model(question: str, df: pd.DataFrame, api_key: str, model_name: str = "gemini-3.8-flash") -> str:
    # 1. Check for instant conversational responses
    quick = get_quick_response(question)
    if quick:
        return quick

    summary = get_cached_summary(df)
    system_prompt = (
        "You are a solar plant data assistant. "
        "Explain solar energy concepts simply and helpfully. "
        "The dataset has these columns: "
        "timestamp, solar_irradiance_wm2, ambient_temperature_c, "
        "panel_temperature_c, cloud_cover_percent, and power_generation_kw. "
        "Dataset summary statistics:\n"
        f"{summary}\n"
        "Do not invent exact dataset values. If the summary is insufficient, "
        "say that the exact answer is unavailable."
    )

    # 2. Try Google Gemini if key is provided (fast, targeted call without slow loops)
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            combined_prompt = f"{system_prompt}\n\nUser Question: {question}"

            # Fast call: Try user's chosen model via Interactions API first
            try:
                if hasattr(client, "interactions"):
                    res = client.interactions.create(
                        model=model_name,
                        input=question,
                        system_instruction=system_prompt,
                        timeout=12.0
                    )
                    txt = extract_text_from_interaction(res)
                    if txt:
                        return txt
            except Exception:
                pass

            # Fallback to generate_content on selected model
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=combined_prompt
                )
                if res and res.text:
                    return res.text
            except Exception as direct_err:
                # If specific model failed and wasn't gemini-flash-latest, try gemini-flash-latest once
                if model_name != "gemini-flash-latest":
                    try:
                        res = client.models.generate_content(
                            model="gemini-flash-latest",
                            contents=combined_prompt
                        )
                        if res and res.text:
                            return res.text
                    except Exception:
                        pass
                return f"⚠️ **Gemini API Error:** {direct_err}"
        except Exception as e:
            return f"⚠️ **Gemini API Error:** {e}"

    # 2. Try Ollama (for local runs)
    try:
        from ollama import chat
        response = chat(
            model="gemma4:e2b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        return response.message.content
    except Exception:
        pass

    # 3. Informative fallback message
    return (
        "⚠️ **AI Model Connection Unavailable**\n\n"
        "To get AI-generated answers in cloud deployment:\n"
        "1. Get a free API key at [Google AI Studio](https://aistudio.google.com/).\n"
        "2. Enter your key in the sidebar, or set `GEMINI_API_KEY` in Streamlit Cloud Secrets.\n\n"
        "💡 *Tip: You can still query statistical data directly (e.g., 'What is the average power generation?', "
        "'Show max solar irradiance', 'Give summary of cloud cover', etc.)!*"
    )

# Display previous conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask about your solar plant data or energy concepts...")

if question:
    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Answer from the dataset where possible
    data_answer = get_data_answer(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing solar data..."):
            if data_answer is not None:
                answer = data_answer
            else:
                answer = query_ai_model(question, df, active_key, selected_model)

        st.markdown(answer)

    # Save assistant answer
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
