"""
Flask Web App for Agriculture Q&A with Voice Support
- Speech-to-Text: User speaks question
- Text-to-Speech: Answer is read aloud
- Multi-language support with Google Translate
"""

from flask import Flask, render_template, request, jsonify, send_file
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from googletrans import Translator
from gtts import gTTS
import speech_recognition as sr
import os
import tempfile
import warnings
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Initialize Google Translator
translator = Translator()

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# ============================================================================
# LOAD AGRICULTURE Q&A MODEL
# ============================================================================
print("Loading agriculture Q&A model...")
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
finetuned_model_path = "tinyllama-agriculture-qa-final"  # folder next to app.py

# Load tokenizer from the base chat model so we get its chat template
qa_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Make sure padding is set
if qa_tokenizer.pad_token is None:
    qa_tokenizer.pad_token = qa_tokenizer.eos_token

# Fallback: define a simple chat template if missing
if not getattr(qa_tokenizer, "chat_template", None):
    qa_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}User: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "Assistant:"
    )


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)

qa_model = PeftModel.from_pretrained(base_model, finetuned_model_path)
qa_model.to(device)
qa_model.eval()

print(f"✓ Agriculture model loaded on {device}!")


# ============================================================================
# TRANSLATION FUNCTIONS
# ============================================================================
def detect_and_translate_to_english(text):
    """Detect language and translate to English"""
    try:
        detection = translator.detect(text)
        detected_lang = detection.lang
        confidence = detection.confidence
        
        print(f"Detected: {detected_lang} (confidence: {confidence})")
        
        if detected_lang == 'en':
            return text, detected_lang
        
        translation = translator.translate(text, src=detected_lang, dest='en')
        return translation.text, detected_lang
        
    except Exception as e:
        print(f"Translation error: {e}")
        return text, 'en'

def translate_to_language(text, target_lang):
    """Translate English text back to target language"""
    if target_lang == 'en':
        return text
    
    try:
        translation = translator.translate(text, src='en', dest=target_lang)
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# ============================================================================
# SPEECH FUNCTIONS
# ============================================================================
def speech_to_text(audio_file, language='auto'):
    """Convert speech to text using Google Speech Recognition"""
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            
        # Use Google Speech Recognition
        if language == 'auto':
            text = recognizer.recognize_google(audio_data)
        else:
            text = recognizer.recognize_google(audio_data, language=language)
        
        return text, True
    except sr.UnknownValueError:
        return "Could not understand audio", False
    except sr.RequestError as e:
        return f"Speech recognition error: {e}", False
    except Exception as e:
        return f"Error: {str(e)}", False

def text_to_speech(text, language='en'):
    """Convert text to speech using gTTS"""
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_filename = temp_file.name
        temp_file.close()
        
        # Generate speech
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(temp_filename)
        
        return temp_filename, True
    except Exception as e:
        print(f"TTS error: {e}")
        return None, False

# ============================================================================
# Q&A FUNCTION
# ============================================================================
def get_answer(question):
    """Get answer from the fine-tuned model"""
    messages = [{"role": "user", "content": question}]
    
    inputs = qa_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # move tensors to device (cpu or gpu)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = qa_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=qa_tokenizer.eos_token_id,
        )
    
    response = qa_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    
    return response.strip()

# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle text question from user"""
    try:
        data = request.json
        user_query = data.get('question', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Please enter a question'}), 400
        
        # Detect language and translate to English
        english_query, detected_lang = detect_and_translate_to_english(user_query)
        print(f"Original ({detected_lang}): {user_query}")
        print(f"English: {english_query}")
        
        # Get answer from model
        english_answer = get_answer(english_query)
        print(f"Answer (EN): {english_answer}")
        
        # Translate back to original language
        translated_answer = translate_to_language(english_answer, detected_lang)
        print(f"Answer ({detected_lang}): {translated_answer}")
        
        return jsonify({
            'original_question': user_query,
            'detected_language': detected_lang,
            'english_question': english_query,
            'english_answer': english_answer,
            'translated_answer': translated_answer,
            'success': True
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/speech-to-text', methods=['POST'])
def handle_speech_to_text():
    """Convert uploaded audio to text"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save temporarily
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file.save(temp_audio.name)
        temp_audio.close()
        
        # Convert to text
        text, success = speech_to_text(temp_audio.name)
        
        # Clean up
        os.unlink(temp_audio.name)
        
        if success:
            return jsonify({
                'text': text,
                'success': True
            })
        else:
            return jsonify({'error': text, 'success': False}), 400
    
    except Exception as e:
        print(f"Speech-to-text error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/text-to-speech', methods=['POST'])
def handle_text_to_speech():
    """Convert text to speech audio"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate speech
        audio_file, success = text_to_speech(text, language)
        
        if success and audio_file:
            return send_file(
                audio_file,
                mimetype='audio/mpeg',
                as_attachment=False,
                download_name='answer.mp3'
            )
        else:
            return jsonify({'error': 'Failed to generate speech'}), 500
    
    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌾 Agriculture Q&A App with Voice Support")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
