import os, time, json, hashlib, re, io
import requests, feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import redis

# =============== ENV ===============
BOT_TOKEN     = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_IDS      = [c.strip() for c in os.getenv("CHAT_IDS", "@news_iran_daily").split(",") if c.strip()]
REDIS_URL     = os.getenv("REDIS_URL", "").strip()          # Upstash: rediss://...
HF_API_KEY    = os.getenv("HF_API_KEY", "").strip()         # اختیاری
LOGO_URL      = os.getenv("LOGO_URL", "").strip()           # اختیاری: PNG شفاف
POLL_SECONDS  = int(os.getenv("POLL_SECONDS", "120"))       # فاصله چک در حالت loop
MAX_DAILY     = int(os.getenv("MAX_POSTS_PER_DAY", "10"))   # سقف روزانه
FRESH_MINUTES = int(os.getenv("FRESH_MINUTES", "240"))      # فقط خبرهای N دقیقه اخیر
HEADER_PREFIX = os.getenv("HEADER_PREFIX", "📰 تازه‌ترین‌ها — @news_iran_daily")
RUN_MODE      = os.getenv("RUN_MODE", "loop").lower()       # loop | oneshot

USER_AGENT    = "Mozilla/5.0 (TelegramNewsBot)"

# RSS: داخلی + فارسی بین‌الملل + خارجی
RSS_FEEDS = [
    # داخلی (اگر فیدی از دسترس خارج شد، موقتاً حذف کن)
    "https://www.isna.ir/rss",
    "https://www.farsnews.ir/rss",
    "https://www.khabaronline.ir/rss",
    "https://www.irna.ir/rss",
    "https://www.mehrnews.com/rss",

    # فارسی بین‌الملل
    "https://feeds.bbci.co.uk/persian/rss.xml",
    "https://parsi.euronews.com/rss",
    "https://rss.dw.com/rdf/rss-fa-all",

    # خارجی (انگلیسی)
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://edition.cnn.com/rss/edition.rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
]

# =============== Redis (Upstash) ===============
if not REDIS_URL:
    print("❌ REDIS_URL تنظیم نشده.")
rds = redis.from_url(REDIS_URL, decode_responses=True, ssl=REDIS_URL.startswith("rediss://"))

def k_posted(h): return f"posted:{h}"              # هش خبرهای ارسال‌شده
def k_daycount(day): return f"count:{day}"         # شمارنده روزانه

# =============== Helpers ===============
def clean_html(s: str) -> str:
    return BeautifulSoup(s or "", "html.parser").get_text().strip()

def is_persian(s: str) -> bool:
    return any('\u0600' <= c <= '\u06FF' for c in s or "")

def first_sentences(txt: str, max_chars=300, max_sents=2) -> str:
    txt = re.sub(r"\s+", " ", (txt or "").strip())
    parts = re.split(r"(?<=[.!؟\?])\s+", txt)
    out = " ".join(parts[:max_sents]) or txt[:max_chars]
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "…"
    return out

# =============== Hugging Face (اختیاری) ===============
def hf_request(model: str, payload: dict) -> dict | None:
    if not HF_API_KEY:
        return None
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json=payload, timeout=40
        )
        if r.status_code == 200:
            return r.json()
        else:
            print("HF error:", r.status_code, r.text[:200])
    except Exception as e:
        print("HF exception:", e)
    return None

def translate_en_to_fa(text: str) -> str:
    if not text: return text
    if not HF_API_KEY: 
        return text  # بدون کلید: ترجمه انجام نمی‌دهیم
    data = hf_request("Helsinki-NLP/opus-mt-en-fa", {"inputs": text})
    try:
        return data[0]["translation_text"]
    except:
        return text

def summarize_text(text: str) -> str:
    if not text: return text
    if not HF_API_KEY:
        return first_sentences(text, max_chars=260, max_sents=2)
    data = hf_request("facebook/bart-large-cnn", {"inputs": text[:2000]})
    try:
        return data[0]["summary_text"]
    except:
        return first_sentences(text, max_chars=260, max_sents=2)

# =============== دسته‌بندی ساده ===============
CATEGORIES = {
    "سیاسی":   ["سیاست","انتخابات","مجلس","دولت","وزیر","رئیس‌جمهور","پارلمان"],
    "اقتصادی": ["اقتصاد","بورس","دلار","تورم","بانک","تجارت","نفت","طلا"],
    "ورزشی":   ["ورزش","فوتبال","والیبال","بسکتبال","تیم","بازی","مسابقه","جام"],
    "علمی":     ["علم","فناوری","تکنولوژی","هوش","فضا","پژوهش","دانشگاه"],
    "اجتماعی": ["حوادث","جامعه","سلامت","آموزش","مدرسه","امنیت"],
    "بین‌الملل":["جهان","بین‌الملل","آمریکا","اروپا","سازمان ملل","ناتو"]
}
def categorize(title_fa: str, summary_fa: str) -> str:
    text = f"{title_fa} {summary_fa}"
    for cat, kw in CATEGORIES.items():
        for k in kw:
            if k in text:
                return cat
    return "عمومی"

# =============== Image helpers ===============
def fetch_bytes(url: str, timeout=25) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200:
            return r.content
    except Exception as e:
        print("Fetch error:", e)
    return None

def find_image(entry) -> str | None:
    if hasattr(entry, "media_content") and entry.media_content:
        u = entry.media_content[0].get("url");   if u: return u
    if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
        u = entry.media_thumbnail[0].get("url"); if u: return u
    if hasattr(entry, "links"):
        for l in entry.links:
            t = (l.get("type") or "").lower()
            if t.startswith("image") or "image" in t or l.get("rel") == "enclosure":
                href = l.get("href")
                if href: return href
    img = getattr(entry, "image", None)
    if isinstance(img, dict) and img.get("href"):
        return img["href"]
    return None

def add_watermark(img_bytes: bytes, logo_bytes: bytes, opacity=0.85, scale=0.22, margin=16) -> bytes:
    try:
        from PIL import Image
        base = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")

        new_w = int(base.width * scale)
        ratio = new_w / logo.width
        logo = logo.resize((new_w, int(logo.height * ratio)))
        if opacity < 1.0:
            alpha = logo.split()[-1]
            alpha = alpha.point(lambda p: int(p * opacity))
            logo.putalpha(alpha)
        x = base.width - logo.width - margin
        y = base.height - logo.height - margin
        canvas = base.copy()
        canvas.alpha_composite(logo, dest=(x, y))
        out = io.BytesIO()
        canvas.convert("RGB").save(out, format="JPEG", quality=90)
        return out.getvalue()
    except Exception as e:
        print("Watermark error:", e)
        return img_bytes

# =============== Telegram ===============
def send_to_channels(caption_html: str, image_url: str | None):
    base = f"https://api.telegram.org/bot{BOT_TOKEN}"
    logo_bytes = fetch_bytes(LOGO_URL) if LOGO_URL else None

    for ch in CHAT_IDS:
        try:
            if image_url:
                photo_raw = fetch_bytes(image_url)
                if photo_raw and logo_bytes:
                    photo_raw = add_watermark(photo_raw, logo_bytes)
                if photo_raw:
                    files = {"photo": ("image.jpg", photo_raw, "image/jpeg")}
                    data  = {"chat_id": ch, "caption": caption_html, "parse_mode": "HTML", "disable_web_page_preview": False}
                    r = requests.post(f"{base}/sendPhoto", data=data, files=files, timeout=30)
                else:
                    r = requests.post(f"{base}/sendMessage", data={"chat_id": ch, "text": caption_html, "parse_mode": "HTML"}, timeout=30)
            else:
                r = requests.post(f"{base}/sendMessage", data={"chat_id": ch, "text": caption_html, "parse_mode": "HTML"}, timeout=30)
            print(f"[{ch}] {r.status_code} {r.text[:140]}")
            time.sleep(1.0)
        except Exception as e:
            print(f"Send error for {ch}:", e)

# =============== News logic ===============
def make_hash(title: str, link: str) -> str:
    return hashlib.sha256(f"{title}|{link}".encode("utf-8")).hexdigest()

def entry_dt(entry):
    dt = None
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
        dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    return dt

def build_caption(title_fa: str, summary_fa: str, link: str, category: str) -> str:
    def esc(s):
        return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    t = esc(title_fa); s = esc(summary_fa); l = esc(link); c = esc(category.replace(" ","_"))
    parts = [
        HEADER_PREFIX,
        f"🏷️ #{c}",
        f"🗞️ <b>{t}</b>",
        f"📝 {s}",
        f"🔗 منبع: <a href=\"{l}\">{l}</a>" if l else ""
    ]
    return "\n".join([p for p in parts if p]).strip()

def process_entry(entry) -> bool:
    title = clean_html(getattr(entry, "title", "") or "")
    link  = getattr(entry, "link", "") or ""
    summ  = clean_html(getattr(entry, "summary", "") or "")

    if not title and not link:
        return False

    # تازگی
    dt = entry_dt(entry)
    if dt:
        age_min = (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
        if age_min > FRESH_MINUTES:
            return False

    h = make_hash(title, link)
    if rds.get(k_posted(h)):
        return False  # قبلاً ارسال شده

    body = summ or title

    # ترجمه/خلاصه
    if is_persian(title + " " + body):
        title_fa = title
        summary_fa = summarize_text(body)  # با HF یا سبک
    else:
        title_fa = translate_en_to_fa(title) if title else title
        summary_fa = summarize_text(body)
        # اگر هنوز انگلیسی بود و HF نداشتیم، همان خلاصه انگلیسی می‌ماند

    # ایموجی ساده برای تیتر
    key = f"{title_fa} {summary_fa}"
    if any(w in key for w in ["فوری","هشدار","انفجار","زلزله","سیل","آتش"]):   prefix = "⚡"
    elif any(w in key for w in ["اقتصاد","دلار","بورس","تورم","بانک"]):        prefix = "💰"
    elif any(w in key for w in ["فوتبال","ورزش","تیم","بازی","مسابقه"]):       prefix = "⚽"
    elif any(w in key for w in ["علم","فناوری","هوش","فضا","پژوهش"]):          prefix = "🤖"
    else: prefix = "🗞️"
    title_fa = f"{prefix} {title_fa}"

    # دسته‌بندی
    category = categorize(title_fa, summary_fa)

    # کپشن
    caption = build_caption(title_fa, summary_fa, link, category)

    # تصویر
    img_url = find_image(entry)

    # محدودیت روزانه
    day = datetime.utcnow().strftime("%Y-%m-%d")
    cnt = int(rds.get(k_daycount(day)) or "0")
    if cnt >= MAX_DAILY:
        print("⏸️ سقف روزانه پر شده.")
        return False

    # ارسال
    send_to_channels(caption, img_url)

    # علامت‌گذاری (۷ روز)
    rds.set(k_posted(h), "1", ex=7*24*3600)
    rds.set(k_daycount(day), str(cnt+1), ex=3*24*3600)
    return True

def run_once():
    sent_now = 0
    headers = {"User-Agent": USER_AGENT}
    for feed in RSS_FEEDS:
        try:
            d = feedparser.parse(feed, request_headers=headers)
            for entry in d.entries[:12]:
                ok = process_entry(entry)
                if ok:
                    sent_now += 1
                    print("✅ پست شد:", getattr(entry, "title", "")[:90])
                    time.sleep(2)
        except Exception as e:
            print("Feed error:", feed, e)
    if sent_now == 0:
        print("ℹ️ خبر جدیدی در بازهٔ تعیین‌شده نبود.")

def main_loop():
    while True:
        try:
            run_once()
        except Exception as e:
            print("Run error:", e)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    if not BOT_TOKEN:
        print("❌ TELEGRAM_TOKEN خالی است."); raise SystemExit
    if not CHAT_IDS:
        print("❌ CHAT_IDS خالی است."); raise SystemExit
    if not REDIS_URL:
        print("❌ REDIS_URL خالی است."); raise SystemExit

    if RUN_MODE == "oneshot":
        run_once()
    else:
        main_loop()
