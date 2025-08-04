# Gerekli kütüphaneleri içe aktarma
import os
import warnings
import tempfile
import torch
import whisper
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, effects
from moviepy import AudioFileClip
from pyannote.audio import Pipeline

# --- TEMEL AYARLAR VE KONTROLLER ---

# Uyarıları bastır (Gereksiz konsol çıktılarını engeller)
warnings.filterwarnings("ignore", category=UserWarning)

# Cihaz seçimi: Mümkünse CUDA destekli GPU, değilse CPU kullan
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Kullanılan cihaz: {DEVICE}")

# İşlenecek video dosyasının yolu
video_file = "mulakat_turk.mp4"  # <<< BURAYA KENDİ VİDEO DOSYANIZIN YOLUNU YAZIN

# Hugging Face token'ınızı buraya girin (pyannote için gerekli)
# https://huggingface.co/settings/tokens adresinden alabilirsiniz.
HF_TOKEN = "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"  # <<< KENDİ TOKEN'INIZI GİRİN

# Video dosyasının varlığını kontrol et
if not os.path.exists(video_file):
    print(f"Hata: '{video_file}' dosyası bulunamadı! Lütfen dosya yolunu kontrol edin.")
    exit()


# --- SES İYİLEŞTİRME FONKSİYONU ---

def enhance_audio(input_path):
    """
    Verilen ses dosyasını bir dizi adımdan geçirerek kalitesini artırır.
    Adımlar: Standardizasyon, Normalizasyon, Filtreleme, Gürültü Azaltma.
    """
    print("Ses iyileştirme süreci başlatılıyor...")
    try:
        # 1. Standardizasyon: Sesi pydub ile yükle, mono yap ve 16kHz'e ayarla
        print("  - Adım 1: Standardizasyon (Mono & 16kHz Resampling)...")
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        # 2. Normalizasyon: Sesi standart bir seviyeye getir
        print("  - Adım 2: Normalizasyon...")
        normalized_audio = effects.normalize(audio)

        # 3. Filtreleme: İnsan sesi frekans aralığı için band-pass filtresi uygula
        print("  - Adım 3: Filtreleme (Band-pass)...")
        filtered_audio = normalized_audio.high_pass_filter(300).low_pass_filter(3400)

        # 4. Gürültü Azaltma: noisereduce kütüphanesi için sesi numpy dizisine çevir
        print("  - Adım 4: Gürültü Azaltma...")
        # Pydub ses segmentini float32 numpy dizisine dönüştür
        samples = np.array(filtered_audio.get_array_of_samples()).astype(np.float32)
        # Örnekleri -1.0 ile 1.0 arasına ölçeklendir
        samples /= np.iinfo(filtered_audio.sample_width * 8).max

        # Gürültüyü azalt
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=16000, stationary=False, prop_decrease=0.85)

        # Temizlenmiş sesi geçici bir dosyaya kaydet
        cleaned_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(cleaned_audio_path, reduced_noise_samples, 16000)

        print(f"Ses iyileştirme tamamlandı. Temizlenmiş dosya: {cleaned_audio_path}")
        return cleaned_audio_path

    except Exception as e:
        print(f"Ses iyileştirme sırasında bir hata oluştu: {e}")
        return None


# --- ANA İŞ AKIŞI ---

raw_audio_file = None
enhanced_audio_file = None

try:
    # 1. VİDEODAN SESİ ÇIKAR
    print("1/5: Videodan ham ses dosyası çıkarılıyor...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        raw_audio_file = temp_audio.name
        video = AudioFileClip(video_file)
        # Sesi çıkarırken doğrudan 16kHz olarak ayarlayalım
        video.write_audiofile(raw_audio_file, codec='pcm_s16le', fps=16000)
        video.close()
    print(f"Ham ses dosyası oluşturuldu: {raw_audio_file}")

    # 2. SESİ İYİLEŞTİR
    print("\n2/5: Ses kalitesi artırılıyor...")
    enhanced_audio_file = enhance_audio(raw_audio_file)
    if not enhanced_audio_file:
        raise Exception("Ses iyileştirme başarısız oldu.")

    # 3. MODELLERİ YÜKLE
    print("\n3/5: AI modelleri yükleniyor...")

    # Whisper: Metne dönüştürme modeli
    # 'base' yerine 'medium' veya 'large' modelleri daha yavaş ama çok daha doğru sonuçlar verir.
    print("   - Whisper 'medium' modeli yükleniyor...")
    whisper_model = whisper.load_model("medium", device=DEVICE)
    print("   - Whisper modeli yüklendi.")

    # Pyannote: Konuşmacı ayırma modeli
    print("   - pyannote.audio pipeline'ı yükleniyor...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    pipeline.to(DEVICE)
    print("   - pyannote.audio pipeline'ı yüklendi.")

    # 4. KONUŞMACI AYRIMI (DIARIZATION)
    print("\n4/5: Konuşmacı ayrımı (Diarization) yapılıyor...")
    diarization = pipeline(enhanced_audio_file)
    print("Diarization tamamlandı.")

    # 5. METNE DÖNÜŞTÜRME (TRANSCRIPTION)
    print("\n5/5: Metne dönüştürme (Transcription) yapılıyor...")
    transcription_result = whisper_model.transcribe(
        enhanced_audio_file,
        word_timestamps=True,
        fp16=torch.cuda.is_available(),
        language='tr'
    )
    print("Transcription tamamlandı.")

    # --- SONUÇLARI BİRLEŞTİRME VE GÖSTERME ---
    print("\n--- KONUŞMA DÖKÜMÜ ---")

    all_words = []
    if 'segments' in transcription_result:
        for segment in transcription_result['segments']:
            if 'words' in segment:
                all_words.extend(segment['words'])

    # Her kelimenin hangi konuşmacıya ait olduğunu bul
    for word in all_words:
        word_center = (word['start'] + word['end']) / 2
        speaker_label = "BİLİNMEYEN"
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= word_center < turn.end:
                speaker_label = speaker
                break
        word['speaker'] = speaker_label

    # Konuşmacı değişikliklerine göre metni grupla ve yazdır
    if all_words:
        current_speaker = all_words[0]['speaker']
        current_sentence = ""

        for i, word_info in enumerate(all_words):
            word_text = word_info['word']

            if word_info['speaker'] == current_speaker:
                current_sentence += word_text
            else:
                if current_sentence.strip():
                    print(f"[{current_speaker}]:{current_sentence.strip()}")

                current_speaker = word_info['speaker']
                current_sentence = word_text

            # Son kelime ise, mevcut cümleyi de yazdır
            if i == len(all_words) - 1:
                if current_sentence.strip():
                    print(f"[{current_speaker}]:{current_sentence.strip()}")

    speakers = {word['speaker'] for word in all_words if 'speaker' in word}
    print(f"\nToplam {len(speakers)} farklı konuşmacı tespit edildi: {', '.join(sorted(list(speakers)))}")


except Exception as e:
    print(f"\nAna işlem akışında bir hata oluştu: {e}")

finally:
    # Geçici dosyaları temizle
    print("\nGeçici dosyalar siliniyor...")
    if raw_audio_file and os.path.exists(raw_audio_file):
        try:
            os.remove(raw_audio_file)
            print(f"  - Silindi: {raw_audio_file}")
        except OSError as e:
            print(f"  - Silinemedi (muhtemelen kullanımda): {raw_audio_file}, Hata: {e}")

    if enhanced_audio_file and os.path.exists(enhanced_audio_file):
        try:
            os.remove(enhanced_audio_file)
            print(f"  - Silindi: {enhanced_audio_file}")
        except OSError as e:
            print(f"  - Silinemedi (muhtemelen kullanımda): {enhanced_audio_file}, Hata: {e}")