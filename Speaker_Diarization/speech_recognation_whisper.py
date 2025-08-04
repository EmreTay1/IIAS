from moviepy import VideoFileClip
import speech_recognition as sr
import os
import spacy
import re

# --- 1. Videodan sesi çıkar ---
video = VideoFileClip("video.mp4")
video.audio.write_audiofile("gecici_ses.wav")

# --- 2. SpeechRecognition ile sesi yazıya çevir ---
r = sr.Recognizer()
with sr.AudioFile("gecici_ses.wav") as source:
    audio = r.record(source)

try:
    metin = r.recognize_google(audio, language="tr-TR")
    print("Yazıya Çevrilen Metin:\n")
    print(metin)
except sr.UnknownValueError:
    print("Ses anlaşılamadı.")
except sr.RequestError as e:
    print(f"Google API hatası: {e}")

# --- 3. Geçici dosya silme ---
os.remove("gecici_ses.wav")


# --- 4. Metni konuşmacılara ayırma ---
def konusmacilari_ayir(metin):
    # Türkçe için spaCy modelini yükle
    try:
        nlp = spacy.load("xx_ent_wiki_sm")  # Çok dilli model
        # Sentencizer bileşenini ekle
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    except Exception as e:
        print(f"spaCy modeli yüklenemedi veya hata oluştu: {e}")
        print("Lütfen 'pip install spacy' ve 'python -m spacy download xx_ent_wiki_sm' komutlarını çalıştırın.")
        return

    # Metni cümlelere böl
    doc = nlp(metin)
    cumleler = [sent.text.strip() for sent in doc.sents]

    # Konuşmacıları saklamak için liste
    konusmacilar = []
    mevcut_konusmaci = None
    konusmaci_sayaci = 1  # Varsayılan konuşmacı ID'leri için

    # Regex ile konuşmacı etiketlerini kontrol et (örneğin, "Ahmet: Merhaba")
    regex_pattern = r"(\w+):\s*(.+)"

    for cumle in cumleler:
        match = re.match(regex_pattern, cumle)
        if match:
            # Eğer konuşmacı etiketi varsa, direkt kullan
            konusmaci, diyalog = match.groups()
            konusmacilar.append({"konusmaci": konusmaci, "diyalog": diyalog})
            mevcut_konusmaci = konusmaci
        else:
            # Etiketsiz cümleler için bağlamsal varsayım
            # Basit bir strateji: Konuşmacılar sırayla konuşuyor
            if mevcut_konusmaci is None:
                mevcut_konusmaci = f"Konuşmacı_{konusmaci_sayaci}"
                konusmaci_sayaci += 1
            konusmacilar.append({"konusmaci": mevcut_konusmaci, "diyalog": cumle})
            # Bir sonraki cümle için konuşmacıyı değiştir (varsayım: sırayla konuşma)
            mevcut_konusmaci = f"Konuşmacı_{konusmaci_sayaci}" if mevcut_konusmaci == f"Konuşmacı_{konusmaci_sayaci - 1}" else f"Konuşmacı_{konusmaci_sayaci - 1}"

    # Sonuçları yazdır
    print("\nKonuşmacılara Ayrılmış Metin:\n")
    for entry in konusmacilar:
        print(f"{entry['konusmaci']}: {entry['diyalog']}")


# Metni konuşmacılara ayır
konusmacilari_ayir(metin)