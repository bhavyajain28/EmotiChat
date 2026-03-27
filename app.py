import warnings
warnings.filterwarnings("ignore")

import os
import json
import streamlit as st
import joblib
import threading
import re
from collections import Counter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ================= LOAD EMOTION MODEL (cached) =================
@st.cache_resource
def load_emotion_model():
    return joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

pipe_lr = load_emotion_model()

# ================= OFFLINE LLM (OLLAMA) =================
@st.cache_resource
def load_llm():
    return OllamaLLM(
        model="phi3mini",
        temperature=0.2,
        num_predict=50,
        num_ctx=256,
        num_thread=os.cpu_count() or 4,
        repeat_penalty=1.1,
        stop=["<|user|>", "<|system|>", "User:", "Bot:", "metadata", "dedicated"]
    )

llm = load_llm()

# ================= EMBEDDINGS (cached) =================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embeddings = load_embeddings()

# ================= LOAD KNOWLEDGE BASE FROM FILE =================
def load_knowledge_base(path: str = "knowledge_base.json") -> list:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return [
        Document(
            page_content=entry["content"],
            metadata={"emotion": entry.get("emotion", "general")}
        )
        for entry in entries
    ]

# ================= VECTOR STORE (disk cached) =================
FAISS_INDEX_PATH = "./faiss_index"

@st.cache_resource
def load_vectorstore():
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        docs = load_knowledge_base("knowledge_base.json")
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(FAISS_INDEX_PATH)
        return vs

# ================= INDIA HELPLINES =================
HELPLINES = {
    "suicide": {
        "title": "🆘 Suicide & Self-Harm Helplines",
        "contacts": [
            {"name": "iCall (TISS)",           "number": "9152987821",    "note": "Mon–Sat, 8am–10pm"},
            {"name": "Vandrevala Foundation",  "number": "18602662345",   "note": "24/7, Free"},
            {"name": "SNEHI",                  "number": "04424640050",   "note": "Mon–Sat, 8am–10pm"},
            {"name": "Emergency / Ambulance",  "number": "112",           "note": "24/7"},
        ]
    },
    "mental_health": {
        "title": "🧠 Mental Health Crisis Helplines",
        "contacts": [
            {"name": "Vandrevala Foundation",  "number": "18602662345",   "note": "24/7, Free"},
            {"name": "iCall (TISS)",           "number": "9152987821",    "note": "Mon–Sat, 8am–10pm"},
            {"name": "NIMHANS",                "number": "08046110007",   "note": "Mental health support"},
            {"name": "Fortis Stress Helpline", "number": "8376804102",    "note": "24/7"},
        ]
    },
    "domestic_violence": {
        "title": "🏠 Domestic Violence Helplines",
        "contacts": [
            {"name": "Women Helpline",         "number": "1091",          "note": "24/7, Free"},
            {"name": "Police",                 "number": "100",           "note": "24/7 Emergency"},
            {"name": "NCW Helpline",           "number": "7827170170",    "note": "National Commission for Women"},
            {"name": "iCall (TISS)",           "number": "9152987821",    "note": "Counseling support"},
        ]
    },
    "substance_abuse": {
        "title": "💊 Substance Abuse & Addiction Helplines",
        "contacts": [
            {"name": "National Drug Helpline", "number": "18001100031",   "note": "24/7, Free, Toll-free"},
            {"name": "NIMHANS De-addiction",   "number": "08046110007",   "note": "Professional support"},
            {"name": "Vandrevala Foundation",  "number": "18602662345",   "note": "24/7, Free"},
            {"name": "iCall (TISS)",           "number": "9152987821",    "note": "Counseling support"},
        ]
    },
}

# ================= EMERGENCY KEYWORD DETECTION =================
EMERGENCY_KEYWORDS = {
    "suicide": [
        "suicide", "suicidal", "kill myself", "end my life", "want to die",
        "marna chahta", "marna chahti", "mar jaunga", "mar jaungi",
        "jaan dena", "jaan de du", "khatam kar lu", "khatam kar lun",
        "nahi rehna", "jeena nahi", "zindagi khatam", "khud ko hurt",
        "self harm", "cut myself", "khud ko kaat", "neend ki goli",
    ],
    "mental_health": [
        "breakdown", "losing my mind", "can't go on", "hopeless", "worthless",
        "no reason to live", "everything is pointless", "mental breakdown",
        "dimag kharab", "pagal ho raha", "pagal ho rahi", "toot gaya", "toot gayi",
        "bahut zyada dard", "koi ummeed nahi", "kuch nahi bacha",
        "crisis", "panic attack", "attack aa raha",
    ],
    "domestic_violence": [
        "hit me", "beating me", "abuse", "abusing", "husband beats",
        "wife beats", "marta hai", "marti hai", "maar raha", "maar rahi",
        "domestic violence", "ghar mein maar", "peet raha", "peet rahi",
        "physical abuse", "marpit", "torture kar raha", "torture kar rahi",
    ],
    "substance_abuse": [
        "drug addiction", "alcohol addiction", "nasha", "nashe mein",
        "drugs le raha", "sharab pi raha", "nahi choot raha", "addiction",
        "overdose", "pills lene ki aadat", "rehab", "withdrawal",
        "neend ki goli zyada", "dawa zyada le li",
    ],
}

def detect_emergency(text: str):
    text_lower = text.lower()
    detected = []
    for category, keywords in EMERGENCY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(category)
    return detected

# ================= RENDER HELPLINE CARD (native Streamlit) =================
def render_helpline_card(category_key: str):
    info = HELPLINES[category_key]
    st.markdown(f"#### {info['title']}")
    for c in info["contacts"]:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"**{c['name']}**  \n_{c['note']}_")
        with col2:
            st.code(c["number"], language=None)
    st.divider()

# ================= OFF-TOPIC GUARD =================
OFF_TOPIC_KEYWORDS = [
    "recipe", "food", "pizza", "cook", "cooking", "movie", "film",
    "cricket", "news", "sport", "game", "weather", "stock", "price",
    "politics", "math", "code", "programming", "history", "science",
    "khana", "khaana", "khelo", "gaana", "music", "song",
]

def is_off_topic(text: str) -> bool:
    return any(word in text.lower() for word in OFF_TOPIC_KEYWORDS)

# ================= HINGLISH MAPPING =================
hinglish_mapping = {
    "khush": "joy", "khushi": "joy", "mast": "joy", "badhiya": "joy",
    "acha": "joy", "accha": "joy", "maza": "joy", "maja": "joy",
    "excited": "joy", "happy": "joy", "pyaar": "joy", "love": "joy",
    "dukhi": "sadness", "sad": "sadness", "rona": "sadness", "ro": "sadness",
    "rone": "sadness", "takleef": "sadness", "dard": "sadness",
    "thaka": "sadness", "thak": "sadness", "akela": "sadness",
    "bura": "sadness", "tanha": "sadness", "upset": "sadness",
    "gussa": "anger", "angry": "anger", "krodh": "anger", "chidchida": "anger",
    "dar": "fear", "darta": "fear", "darti": "fear", "scared": "fear",
    "pareshan": "fear", "ghabra": "fear", "mushkil": "fear", "nervous": "fear",
    "sharm": "shame", "sharminda": "shame",
    "shocked": "surprise", "hairan": "surprise",
}

# ================= EMOJIS =================
emojis_list = ["😠","🤮","😨","🤗","😂","😐","😔","😳","😮","💛","😊"]

emoji_emotion_dict = {
    "😠": "anger", "🤮": "disgust", "😨": "fear",
    "🤗": "happy", "😂": "joy", "😐": "neutral",
    "😔": "sadness", "😳": "shame", "😮": "surprise",
    "💛": "happy", "😊": "happy"
}

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨",
    "happy": "🤗", "joy": "😂", "neutral": "😐",
    "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# ================= PROMPT TEMPLATE =================
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["emotion", "emoji", "context", "history", "user_message"],
    template="""<|system|>
You are a mental wellness chatbot ONLY.
If the user asks anything NOT related to emotions or mental health, reply ONLY with: "I can only help with emotions and mental wellness."
Do NOT answer questions about food, recipes, technology, movies, sports, or any other topic.
Reply warmly in 2 sentences only. Never mention metadata, tags, labels, or categories.
Do NOT copy the wellness tip word for word — use it naturally.
Do NOT repeat the chat history, just use it as context.
Emotion detected: {emotion} {emoji}
Wellness tip: {context}
Chat so far: {history}
</s>
<|user|>{user_message}</s>
<|assistant|>"""
)

# ================= EMOTION PREDICTION =================
def predict_emotion(text):
    for e, emo in emoji_emotion_dict.items():
        if e in text:
            return emo
    words = text.lower().split()
    matches = [hinglish_mapping[w] for w in words if w in hinglish_mapping]
    if matches:
        return Counter(matches).most_common(1)[0][0]
    return pipe_lr.predict([text])[0]

# ================= RAG RETRIEVAL =================
def retrieve_context(emotion: str, user_message: str, vectorstore) -> str:
    query = f"{emotion}: {user_message}"
    docs = vectorstore.similarity_search(query, k=1)
    return docs[0].page_content if docs else ""

# ================= HISTORY SUMMARY =================
def build_history_summary(chat_history: list) -> str:
    if not chat_history:
        return "None."
    lines = []
    for role, msg in chat_history[-4:]:
        clean = msg.split(": ", 1)[-1] if ": " in msg else msg
        label = "User" if role == "user" else "Bot"
        lines.append(f"{label}: {clean}")
    return " | ".join(lines)

# ================= CLEAN BOT REPLY =================
def clean_reply(text: str) -> str:
    text = re.sub(r"\([^)]*dedicated[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"\b(metadata|category|tag)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================= OFFLINE AI REPLY =================
def get_ai_reply_offline(user_message: str, emotion: str, chat_history: list, vectorstore) -> str:
    context = retrieve_context(emotion, user_message, vectorstore)
    emoji   = emotions_emoji_dict.get(emotion, "")
    history = build_history_summary(chat_history)
    final_prompt = PROMPT_TEMPLATE.format(
        emotion=emotion, emoji=emoji, context=context,
        history=history, user_message=user_message
    )
    raw_reply = llm.invoke(final_prompt).strip()
    return clean_reply(raw_reply)

# ================= WARMUP =================
def _warmup():
    try:
        llm.invoke("hi")
    except:
        pass

# ================= UI BUBBLES =================
def message_bubble(text, is_bot):
    style = (
        "background:#2f2f2f;color:white;padding:12px;border-radius:12px;"
        "margin:6px;width:75%;border:1px solid #555;"
        if is_bot else
        "background:#075E54;color:white;padding:12px;border-radius:12px;"
        "margin:6px;margin-left:25%;width:75%;text-align:right;"
    )
    st.markdown(f"<div style='{style}'>{text}</div>", unsafe_allow_html=True)

# ================= SEND MESSAGE =================
def process_message(vectorstore):
    msg = st.session_state.user_input.strip()
    if not msg:
        return

    # 1. Emergency check — highest priority
    emergencies = detect_emergency(msg)
    if emergencies:
        emotion = predict_emotion(msg)
        with st.spinner("🤔 Thinking..."):
            bot_reply = get_ai_reply_offline(msg, emotion, st.session_state.chat, vectorstore)
        st.session_state.chat.append(("user", f"🧑 You: {msg}"))
        st.session_state.chat.append(("bot", bot_reply))
        st.session_state.chat.append(("emergency", emergencies))
        st.session_state.user_input = ""
        return

    # 2. Off-topic guard
    if is_off_topic(msg):
        st.session_state.chat.append(("user", f"🧑 You: {msg}"))
        st.session_state.chat.append(("bot", "I'm here to support your emotional wellbeing — I can't help with that topic! 💛"))
        st.session_state.user_input = ""
        return

    # 3. Normal flow
    emotion = predict_emotion(msg)
    with st.spinner("🤔 Thinking..."):
        bot_reply = get_ai_reply_offline(msg, emotion, st.session_state.chat, vectorstore)
    st.session_state.chat.append(("user", f"🧑 You: {msg}"))
    st.session_state.chat.append(("bot", bot_reply))
    st.session_state.user_input = ""

# ================= MAIN =================
def main():
    st.set_page_config("Emotion Chatbot", "🤖", layout="wide")
    st.markdown("""<style>.stApp { overflow-y: auto; }</style>""", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;'>🤖 Emotion-Aware Chatbot</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:gray;'>Powered by Ollama + RAG · Understands your emotions</p>",
        unsafe_allow_html=True
    )

    vectorstore = load_vectorstore()
    threading.Thread(target=_warmup, daemon=True).start()

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    chat_container = st.container()
    with chat_container:
        for item in st.session_state.chat:
            role, content = item[0], item[1]

            if role == "user":
                message_bubble(content, is_bot=False)

            elif role == "bot":
                message_bubble(f"🤖 Bot: {content}", is_bot=True)

            elif role == "emergency":
                # content = list of emergency category keys
                st.warning("⚠️ It seems you may be going through something serious. You are not alone — please reach out for help right away:")
                for cat in content:
                    render_helpline_card(cat)

    cols = st.columns(len(emojis_list))
    for i, e in enumerate(emojis_list):
        with cols[i]:
            if st.button(e, key=f"emoji_{i}"):
                st.session_state.user_input += e

    with st.form("chat_form"):
        st.text_input("Write your message…", key="user_input")
        st.form_submit_button("Send", on_click=lambda: process_message(vectorstore))

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
        st.rerun()

if __name__ == "__main__":
    main()
