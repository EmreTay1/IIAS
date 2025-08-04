# Gerekli kütüphaneleri içe aktar
import warnings
import torch
from pyannote.audio import Pipeline
import os

# --- YENİ EKLENEN BÖLÜM ---
# pyannote.audio'dan kaynaklanan ve çalışmayı engellemeyen UserWarning'leri gizle
# Bu, "degrees of freedom is <= 0" uyarısının görünmesini engeller.
warnings.filterwarnings("ignore", category=UserWarning)
# -------------------------

# -------------------------------------------------------------------
#                         KULLANICI AYARLARI
# Lütfen bu bölümü kendi bilgilerinizle ve dosya yolunuzla güncelleyin.
# -------------------------------------------------------------------

# 1. Hugging Face'ten aldığınız Access Token'ı buraya yapıştırın.
#    Token almak için: https://huggingface.co/settings/tokens
HF_TOKEN = "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"  # LÜTFEN KENDİ TOKEN'INIZI GİRİN

# 2. Konuşmacı ayrımı yapılacak ses dosyasının tam yolunu belirtin.
#    Windows için örnek: "C:\\Users\\Kullanici\\Muzik\\toplanti_kaydi.wav"
#    macOS/Linux için örnek: "/home/kullanici/sesler/toplanti_kaydi.wav"
SES_DOSYASI_YOLU = "video.wav"  # LÜTFEN GEÇERLİ BİR DOSYA YOLU GİRİN

# 3. (Opsiyonel) Ses dosyasındaki konuşmacı sayısını biliyorsanız buraya yazın.
#    Bilinmiyorsa veya otomatik algılanmasını istiyorsanız None olarak bırakın.
#    Örnek: KONUSMACI_SAYISI = 2
KONUSMACI_SAYISI = None

# -------------------------------------------------------------------
#                         KODUN ANA BÖLÜMÜ
#                 Bu bölümü değiştirmenize gerek yoktur.
# -------------------------------------------------------------------

# Dosya yolunun var olup olmadığını kontrol et
if not os.path.exists(SES_DOSYASI_YOLU):
    print(f"HATA: Belirtilen dosya yolu bulunamadı: {SES_DOSYASI_YOLU}")
    print("Lütfen 'SES_DOSYASI_YOLU' değişkenini doğru bir şekilde ayarladığınızdan emin olun.")
    exit()

# Adım 1: Cihazı Belirle (GPU varsa CUDA, yoksa CPU)
try:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hesaplama için kullanılacak cihaz: {DEVICE}")
except Exception as e:
    print(f"Cihaz belirlenirken bir hata oluştu: {e}")
    exit()

# Adım 2: Konuşmacı Ayrımı Pipeline'ını Hugging Face'ten Yükle
try:
    print("Konuşmacı ayrımı modeli (pipeline) yükleniyor... (Bu işlem ilk seferde biraz sürebilir)")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    ).to(DEVICE)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    print(
        "Lütfen Hugging Face token'ınızın doğru olduğundan ve 'pyannote/speaker-diarization-3.1' model koşullarını kabul ettiğinizden emin olun.")
    exit()

# Adım 3: Ses Dosyasını İşle ve Konuşmacı Ayrımını Gerçekleştir
try:
    print(f"'{os.path.basename(SES_DOSYASI_YOLU)}' dosyası üzerinde konuşmacı ayrımı yapılıyor...")

    # Pipeline'a gönderilecek parametreleri hazırla
    pipeline_params = {}
    if KONUSMACI_SAYISI:
        pipeline_params["num_speakers"] = KONUSMACI_SAYISI

    diarization_result = pipeline(SES_DOSYASI_YOLU, **pipeline_params)

    print("İşlem başarıyla tamamlandı.")
except Exception as e:
    print(f"Ses dosyası işlenirken bir hata oluştu: {e}")
    exit()

# Adım 4: Sonuçları Ekrana Yazdır
print("\n--- KONUŞMACI AYRIMI SONUÇLARI ---")
# DÜZELTME: 'Annotation' nesnesinin boş olup olmadığını kontrol etmenin doğru yolu
# doğrudan bir boolean ifadesi olarak kullanmaktır. .is_empty() metodu artık yoktur.
if diarization_result:
    # itertracks metodu ile zaman aralıklarını ve konuşmacı etiketlerini al
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        # 'turn.start' -> başlangıç saniyesi
        # 'turn.end'   -> bitiş saniyesi
        # 'speaker'    -> atanmış konuşmacı etiketi (Örn: SPEAKER_00)
        print(f"Zaman: {turn.start:07.2f}s - {turn.end:07.2f}s  | Konuşmacı: {speaker}")
else:
    print("Ses dosyasında herhangi bir konuşma tespit edilemedi.")