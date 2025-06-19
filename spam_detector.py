import pandas as pd
import gradio as gr
from gtts import gTTS
import tempfile
import joblib

# Load saved ML model
model = joblib.load("spam_model.pkl")

# Keyword rules
spam_keywords = {"kill", "prize", "won", "lottery", "congratulations", "urgent", "free", "click", "money"}

# Languages for voice
language_map = {
    "English": "en",
    "Tamil": "ta",
    "Telugu": "te",
    "Hindi": "hi",
    "Kannada": "kn",
    "Malayalam": "ml"
}

# Prediction function
def predict_message(text, language="English"):
    text_lower = text.lower()
    lang_code = language_map.get(language, "en")

    spam_msg = {
        "English": "Kindly delete the message",
        "Tamil": "தயவுசெய்து செய்தியை நீக்கவும்",
        "Telugu": "దయచేసి సందేశాన్ని తొలగించండి",
        "Hindi": "कृपया संदेश को हटा दें",
        "Kannada": "ದಯವಿಟ್ಟು ಸಂದೇಶವನ್ನು ಅಳಿಸಿ",
        "Malayalam": "ദയവായി സന്ദേശം നീക്കം ചെയ്യുക"
    }
    ham_msg = {
        "English": "Good to have",
        "Tamil": "இது வைத்திருப்பது நல்லது",
        "Telugu": "ఇది ఉండడం మంచిది",
        "Hindi": "रखना अच्छा है",
        "Kannada": "ಇದನ್ನು ಇಟ್ಟುಕೊಳ್ಳುವುದು ಉತ್ತಮ",
        "Malayalam": "ഇത് നല്ലതാണ്"
    }

    if any(keyword in text_lower for keyword in spam_keywords):
        label = "🚫 SPAM (Detected via keyword)"
        voice_response = spam_msg[language]
    else:
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0][pred]
        if pred == 1:
            label = f"🚫 SPAM (Confidence: {prob:.2%})"
            voice_response = spam_msg[language]
        else:
            label = f"📩 HAM (Confidence: {prob:.2%})"
            voice_response = ham_msg[language]

    # Always female voice
    tts = gTTS(text=voice_response, lang=lang_code, slow=False, tld="co.in")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return label, fp.name

# Custom Dark UI CSS
custom_css = """
body, .gradio-container {
    background-color: #111 !important;
    color: #f0f0f0 !important;
}
input, textarea, .gr-button, .gr-box, label, select {
    background-color: #222 !important;
    color: #f0f0f0 !important;
    border: 1px solid #333 !important;
}
.gr-button:hover {
    background-color: #333 !important;
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css, title="Spam Detector") as app:
    gr.HTML("""
    <div style='
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0,0,0,0.3);
        font-family: sans-serif;
    '>
        <h1 style='font-size: 3em;'>📨 Spam Classifier App</h1>
        <p style='font-size: 1.2em;'>An intelligent Spam/Ham detector with multilingual voice support!</p>
    </div>
    """)

    gr.Markdown("### 🔍 Check if a Message is Spam or Ham")

    with gr.Row():
        with gr.Column(scale=1):
            msg_input = gr.Textbox(label="✉️ Your Message", lines=5, placeholder="Type message here...")
            language = gr.Dropdown(
                ["English", "Tamil", "Telugu", "Hindi", "Kannada", "Malayalam"], 
                label="🌐 Language", value="English"
            )
            classify_btn = gr.Button("🧠 Classify Message", variant="primary")
        with gr.Column(scale=1):
            result_box = gr.Textbox(label="📢 Prediction", interactive=False)
            audio_box = gr.Audio(label="🎧 Voice Feedback", type="filepath")

    classify_btn.click(predict_message, inputs=[msg_input, language], outputs=[result_box, audio_box])

    gr.Markdown("""
    <hr>
    <div style='text-align:center; font-size:14px; color: #bbb;'>
        🎓 Developed by <strong>Hemanth</strong> | Powered by Scikit-learn, gTTS & Gradio 🤖
    </div>
    """)

if __name__ == "__main__":
    app.launch()
