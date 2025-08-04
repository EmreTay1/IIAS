from pyannote.audio import Pipeline
import torch
import os
import warnings
import tempfile
from moviepy import AudioFileClip
from pydub import AudioSegment
import whisper  # YENİ: Whisper kütüphanesini içe aktar

# Uyarıları filtrele
warnings.filterwarnings("ignore", category=UserWarning)

# --- Kodunuzun Orijinal Kısmı (Değişiklik Yok) ---

# Cihaz seçimi
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Kullanılan cihaz: {DEVICE}")

# Video dosyası yolu
video_file = "test.mp4"  # Video dosyanızın yolunu buraya yazın

# Video dosyası var mı kontrol et
if not os.path.exists(video_file):
    print(f"Hata: '{video_file}' dosyası bulunamadı! Lütfen dosya yolunu kontrol edin.")
    print(f"Geçerli çalışma dizini: {os.getcwd()}")
    exit()

# Video dosyasından sesi çıkar ve geçici bir .wav dosyası oluştur
try:
    print("Video ses dosyasına dönüştürülüyor...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        audio_file = temp_audio_file.name
        video = AudioFileClip(video_file)
        video.write_audiofile(audio_file, codec='pcm_s16le')  # Whisper için standart format
        video.close()
    print(f"Geçici ses dosyası oluşturuldu: {audio_file}")
except Exception as e:
    print(f"Video-ses dönüşüm hatası: {e}")
    exit()

# Ses dosyasını kontrol et
try:
    audio = AudioSegment.from_file(audio_file)
    print(f"Ses dosyası süresi: {len(audio) / 1000:.2f} saniye")
    if len(audio) < 1000:
        print("Hata: Ses dosyası çok kısa!")
        os.remove(audio_file)
        exit()
except Exception as e:
    print(f"Ses dosyası okuma hatası: {e}")
    os.remove(audio_file)
    exit()

# --- YENİ BÖLÜM: Whisper Modelini Yükleme ---

try:
    # "base" modeli hızlı ve çoğu durum için yeterlidir.
    # Daha yüksek doğruluk için "small", "medium", "large-v2" modellerini kullanabilirsiniz.
    # Model büyüdükçe işlem süresi ve VRAM kullanımı artar.
    print("Whisper modeli yükleniyor ('base')...")
    whisper_model = whisper.load_model("base", device=DEVICE)
    print("Whisper modeli yüklendi.")
except Exception as e:
    print(f"Whisper modeli yükleme hatası: {e}")
    os.remove(audio_file)
    exit()

# --- pyannote Pipeline Yükleme (Orijinal Kod) ---
try:
    print("pyannote.audio pipeline'ı yükleniyor...")
    # ÖNEMLİ: Kendi HuggingFace token'ınızı kullanmaya devam edin.
    # Token'ı koda yazmak yerine environment variable olarak ayarlamak daha güvenlidir.
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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"  # KENDİ TOKEN'INIZI GİRİN
    )
    pipeline.to(DEVICE)
    print("pyannote.audio pipeline'ı yüklendi.")
except Exception as e:
    print(f"Pipeline yükleme hatası: {e}")
    os.remove(audio_file)
    exit()

# Hiperparametreler (Orijinal Kod)
# Bu parametrelerle oynayarak daha iyi sonuçlar elde edebilirsiniz.
params = {
    "segmentation": {
        "threshold": 0.8,
        "min_duration_on": 0.5,
        "min_duration_off": 0.5,
    },
    "clustering": {
        "threshold": 0.8,
    }
}
pipeline.hyperparameters = params

# Diarization (Orijinal Kod)
print("Diarization (konuşmacı ayırma) işlemi başlatılıyor...")
try:
    diarization = pipeline(audio_file)
    print("Diarization işlemi tamamlandı.")
except Exception as e:
    print(f"Diarization hatası: {e}")
    os.remove(audio_file)
    exit()

# --- YENİ BÖLÜM: Ses Dosyasını Metne Dönüştürme (Transcription) ---

print("Transcription (metne dönüştürme) işlemi başlatılıyor...")
try:
    # `word_timestamps=True` kelime bazında zaman damgası almak için kullanılır.
    transcription_result = whisper_model.transcribe(audio_file, word_timestamps=True, fp16=torch.cuda.is_available())
    print("Transcription işlemi tamamlandı.")
except Exception as e:
    print(f"Transcription hatası: {e}")
    os.remove(audio_file)
    exit()

# --- DEĞİŞTİRİLEN BÖLÜM: Sonuçları Birleştirme ve Yazdırma ---

print("\n--- Konuşmacı ve Metin Dökümü ---")

# Whisper'dan gelen kelime segmentlerini daha kolay işlemek için bir listeye alalım
all_words = []
for segment in transcription_result['segments']:
    for word in segment['words']:
        all_words.append(word)

# Diarization sonuçlarını (konuşmacı zaman aralıkları) dolaşalım
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_time = turn.start
    end_time = turn.end

    # Bu zaman aralığına düşen kelimeleri bulalım
    segment_words = [word['word'] for word in all_words if word['start'] >= start_time and word['end'] <= end_time]

    if segment_words:
        # Kelimeleri birleştirerek segmentin metnini oluşturalım
        segment_text = "".join(segment_words).strip()

        # Sadece anlamlı metin içeren segmentleri yazdıralım
        if segment_text:
            start_formatted = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{start_time % 60:05.2f}"
            end_formatted = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{end_time % 60:05.2f}"
            print(f"[{start_formatted} --> {end_formatted}] {speaker}: {segment_text}")

# Konuşmacı sayısı
speakers = set([speaker for _, _, speaker in diarization.itertracks(yield_label=True)])
print(f"\nToplam {len(speakers)} farklı konuşmacı tespit edildi.")

# Geçici ses dosyasını sil (Orijinal Kod)
try:
    os.remove(audio_file)
    print(f"\nGeçici ses dosyası silindi: {audio_file}")
except Exception as e:
    print(f"Geçici dosya silme hatası: {e}")