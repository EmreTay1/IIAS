import cv2
import os
import subprocess
import whisper
from deepface import DeepFace  # deepface kütüphanesini ekledik


# --- Önceki adımlardan gelen fonksiyonlar ---
def extract_audio_from_video(video_path, audio_output_path):
    print(f"'{video_path}' dosyasından ses ayrıştırılıyor...")
    command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y',
               audio_output_path]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Ses başarıyla '{audio_output_path}' dosyasına kaydedildi.")
        return True
    except Exception as e:
        print(f"FFmpeg hatası! {e}")
        return False


def transcribe_audio_with_timestamps(audio_path):
    print("Whisper modeli yükleniyor...")
    model = whisper.load_model("base")
    print("Ses metne dönüştürülüyor...")
    result = model.transcribe(audio_path)
    print("Metne dönüştürme tamamlandı.")
    return result["segments"]


# --- Ana İşlem Fonksiyonu ---
def analyze_multimodal_video_with_emotion(video_path):
    audio_output_path = "temp_audio.wav"

    # --- ADIM 1: Ses Hazırlığı ---
    if not extract_audio_from_video(video_path, audio_output_path):
        return
    text_segments = transcribe_audio_with_timestamps(audio_output_path)
    # --- ADIM 2: Video Analizi (Duygu Tanıma ile) ---
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("\n--- BİRLEŞTİRİLMİŞ MULTIMODAL ANALİZ (DUYGU + METİN) BAŞLIYOR ---")
    print("Not: Her karede duygu analizi yapmak yavaş olabilir. Sabırlı olun...")

    frame_num = 0
    segment_index = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        timestamp = frame_num / fps
        dominant_emotion = "N/A"  # Başlangıçta duygu yok

        # --- YÜZ İFADESİNDEN DUYGU ANALİZİ ---
        # Not: Her karede analiz yapmak sistemi yavaşlatır.
        # Performans için her N'inci karede (örn: every 15 frames) analiz yapabilirsiniz.
        # if frame_num % 15 == 0:
        try:
            # DeepFace.analyze fonksiyonu yüzü bulur ve duyguyu analiz eder.
            # enforce_detection=False parametresi, karede yüz bulamazsa hata vermesini engeller.
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            # DeepFace birden fazla yüz bulursa diye liste döndürür, ilkini alıyoruz.
            if isinstance(analysis, list) and len(analysis) > 0:
                dominant_emotion = analysis[0]['dominant_emotion']
        except Exception as e:
            # Bazen yüz bulunsa bile beklenmedik hatalar olabilir
            dominant_emotion = "Analiz Hatası"
        # --- Konuşulan metni bulma ---
        current_text = "..."
        if segment_index < len(text_segments) and timestamp >= text_segments[segment_index]['start']:
            current_text = text_segments[segment_index]['text']
            if timestamp > text_segments[segment_index]['end']:
                segment_index += 1
                if not (segment_index < len(text_segments) and timestamp >= text_segments[segment_index]['start']):
                    current_text = "..."
        # --- Sonucu Ekrana Yazdırma ---
        # Duyguyu büyük harflerle daha belirgin yapalım
        print(f"Zaman: {timestamp:.2f}s | Duygu: {dominant_emotion.upper():<10} | Konuşma: {current_text.strip()}")
        frame_num += 1

    video.release()
    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)
        print(f"\nGeçici dosya '{audio_output_path}' silindi.")


if __name__ == "__main__":
    # LÜTFEN BU YOLU KENDİ VİDEO DOSYANIZLA DEĞİŞTİRİN
    video_input_path = "Hunharca.mp4"
    if not os.path.exists(video_input_path):
        print(f"HATA: '{video_input_path}' video dosyası bulunamadı.")
    else:
        analyze_multimodal_video_with_emotion(video_input_path)