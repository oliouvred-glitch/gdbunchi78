"""
AI IMAGE EDITOR — FLASK BACKEND (Vercel-ready)
================================================
PIPELINE (from catdog_classifier notebook):
  1. Data Collection     -> /upload + /gallery/save (SQLite gallery database)
  2. Data Preprocessing  -> resize / smooth / grayscale / blur helpers
  3. Feature Engineering -> color_quantize(), extract_features() (HOG + color histogram)
  4. Model Training      -> /train (Logistic Regression on labeled gallery images)
  5. Model Evaluation    -> accuracy + classification report returned by /train
  6. Model Deployment    -> /process + /classify endpoints
"""

import os, io, re, uuid, sqlite3, pickle, base64
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_from_directory

# ---------- optional AI cutout (rembg) ----------
try:
    from rembg import remove, new_session
    REMBG = True
except Exception:
    REMBG = False

# ---------- optional ML (steps 4-6) ----------
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN = True
except Exception:
    SKLEARN = False

app = Flask(__name__)

# ---------- paths (Vercel only allows writing to /tmp) ----------
BASE    = "/tmp" if os.path.isdir("/tmp") else "."
UPLOAD  = os.path.join(BASE, "uploads");  os.makedirs(UPLOAD,  exist_ok=True)
RESULT  = os.path.join(BASE, "results");  os.makedirs(RESULT,  exist_ok=True)
GALLERY = os.path.join(BASE, "gallery");  os.makedirs(GALLERY, exist_ok=True)
DB_PATH    = os.path.join(BASE, "gallery.db")
MODEL_PATH = os.path.join(BASE, "catdog_model.pkl")

ALLOWED = {"png", "jpg", "jpeg", "bmp", "webp"}


# ======================================================================
# KEYWORDS + COLORS
# ======================================================================
COLORS = {
    "red": (255, 0, 0),       "blue": (0, 0, 255),      "green": (0, 128, 0),
    "yellow": (255, 255, 0),  "purple": (128, 0, 128),  "pink": (255, 192, 203),
    "orange": (255, 165, 0),  "brown": (165, 42, 42),   "black": (0, 0, 0),
    "white": (255, 255, 255), "gray": (128, 128, 128),  "grey": (128, 128, 128),
    "light blue": (173, 216, 230), "light green": (144, 238, 144),
}
HUES = {"red": 0, "orange": 30, "yellow": 60, "light green": 90, "green": 120,
        "light blue": 200, "blue": 240, "purple": 270, "pink": 330, "brown": 20}

FRAME_STYLES = {
    "classic": ["classic frame", "classic border", "simple frame"],
    "black":   ["black frame", "black border"],
    "white":   ["white frame", "white border"],
    "gold":    ["gold frame", "gold border", "golden"],
    "red":     ["red frame", "red border"],
    "blue":    ["blue frame", "blue border"],
    "double":  ["double frame", "double border"],
}

KEYWORDS = {
    "remove_background": ["remove background", "background remover", "remove bg", "bg remove",
                          "cut out", "cutout", "cut image", "cut neatly", "cut image neatly",
                          "transparent background", "no background", "erase background"],
    "cut_paste": ["cut and paste", "cut paste", "paste on", "paste onto", "paste into",
                  "paste it on another image", "merge images", "cut image and paste"],
    "white_background": ["white background", "white bg", "make background white", "bg white"],
    "bg_color": ["background color", "background colour", "change background color",
                 "change background colour", "change bg color", "bg colour"],
    "black_white": ["black and white", "black & white", "black white", "grayscale",
                    "greyscale", "bw", "monochrome", "b&w"],
    "cartoon": ["cartoon", "cartoon style", "cartoon image", "comic", "cartoonify"],
    "ghibli": ["ghibli", "studio ghibli", "ghibli style", "anime", "anime style"],
    "frame": ["frame", "border", "add frame", "photo frame", "frame styles", "add border"],
    "favicon": ["favicon", "icon", "favicon size", "make favicon"],
    "upscale": ["upscale", "upscale image", "upscale 2x", "enhance", "hd", "4k", "super resolution"],
    "smooth": ["smooth", "smooth image", "smoothen", "soften", "painterly"],
    "color_quantize": ["color quantize", "color quantized", "quantize", "quantization", "posterize"],
    "shape_album": ["shape album", "shapes album", "shapes", "triangle", "hexagon", "octagon"],
    "ink_outline": ["ink outline", "ink outlines", "ink", "outline", "outlines",
                    "sketch", "pencil", "line art"],
    "three_d": ["3d", "3d image", "3-d", "anaglyph", "3d effect"],
    "two_d": ["2d", "2d image", "2-d", "flat image", "flat 2d"],
    "change_color": ["change color", "change colour", "recolor", "color change",
                     "change image color", "colour change"],
    "text": ["text", "add text", "caption", "add caption", "write text", "text:"],
    "video": ["video", "gif", "animation", "animate", "seconds", "make video"],
    "classify": ["classify", "cat or dog", "detect animal", "what animal", "cat dog", "predict"],
}

# canonical pipeline order
ORDER = ["remove_background", "cut_paste", "white_background", "bg_color", "black_white",
         "cartoon", "ghibli", "smooth", "color_quantize", "change_color", "two_d",
         "ink_outline", "three_d", "shape_album", "frame", "text", "favicon",
         "upscale", "video", "classify"]

FRAME_COLORS = {"classic": (255, 255, 255), "black": (0, 0, 0), "gold": (212, 175, 55),
                "red": (200, 30, 30), "blue": (30, 80, 200), "white": (255, 255, 255)}


# ======================================================================
# HELPERS  (Step 2: Data Preprocessing)
# ======================================================================
def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED


def to_pil(img):
    """cv2 BGR/BGRA -> PIL"""
    if isinstance(img, Image.Image):
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    if img.ndim == 3:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(img)


def to_bgr(img):
    """PIL / RGBA / BGRA -> cv2 BGR"""
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS gallery(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL, name TEXT, prompt TEXT,
        created TEXT DEFAULT CURRENT_TIMESTAMP)""")
    return conn


def load_image(file_id):
    """Load from uploads folder OR from gallery DB ('g:<id>')."""
    for f in os.listdir(UPLOAD):
        if f.startswith(file_id):
            return Image.open(os.path.join(UPLOAD, f)).convert("RGB")
    if file_id.startswith("g:"):
        try:
            conn = get_db()
            row = conn.execute("SELECT filename FROM gallery WHERE id=?",
                               (int(file_id[2:]),)).fetchone()
            conn.close()
            if row:
                return Image.open(os.path.join(GALLERY, row[0])).convert("RGB")
        except Exception:
            pass
    return None


# ======================================================================
# FEATURES  (Step 3: Feature Engineering)
# ======================================================================
def remove_background(img):
    """Neatly cut the subject -> RGBA with transparent background."""
    pil = to_pil(img).convert("RGB")
    if REMBG:
        try:
            return remove(pil, session=new_session("isnet-general-use"),
                          alpha_matting=True,
                          alpha_matting_foreground_threshold=240,
                          alpha_matting_background_threshold=10,
                          alpha_matting_erode_size=10)
        except Exception:
            pass
    # Fallback: OpenCV GrabCut
    cv_img = to_bgr(pil)
    h, w = cv_img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mx, my = max(2, w // 15), max(2, h // 15)
    rect = (mx, my, max(3, w - 2 * mx), max(3, h - 2 * my))
    try:
        cv2.grabCut(cv_img, mask, rect, np.zeros((1, 65), np.float64),
                    np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return pil.convert("RGBA")
    m2 = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    m2 = cv2.GaussianBlur(m2, (5, 5), 0)                       # smooth edges
    return to_pil(np.dstack([cv_img, m2]))


def cutout_on_color(img, rgb):
    """Neatly cut the image and paste it on a solid color background."""
    cut = remove_background(img)
    bg = Image.new("RGBA", cut.size, tuple(rgb) + (255,))
    bg.alpha_composite(cut)
    return bg.convert("RGB")


def paste_on_image(fg, bg, scale=0.55):
    """Neatly cut the first image and paste it neatly onto another image."""
    cut = remove_background(fg)
    canvas = bg.convert("RGBA")
    fw = int(canvas.width * scale)
    fh = max(1, int(cut.height * fw / max(1, cut.width)))
    cut = cut.resize((fw, fh), Image.LANCZOS)
    if (fw, fh) > canvas.size:
        canvas = canvas.resize((fw, fh), Image.LANCZOS)
    canvas.alpha_composite(cut, ((canvas.width - fw) // 2, (canvas.height - fh) // 2))
    return canvas.convert("RGB")


def to_black_white(img):
    g = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def cartoon_effect(img):
    cv_img = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2RGB)
    for _ in range(7):
        cv_img = cv2.bilateralFilter(cv_img, 9, 9, 7)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 7), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    out = cv2.bitwise_and(cv_img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def ghibli_effect(img):
    cv_img = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2RGB)
    for _ in range(5):
        cv_img = cv2.bilateralFilter(cv_img, 9, 75, 75)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    Z = cv_img.reshape(-1, 3).astype(np.float32)
    _, label, center = cv2.kmeans(Z, 8, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS)
    cv_img = np.uint8(center)[label.flatten()].reshape(cv_img.shape)
    cv_img = cv2.bilateralFilter(cv_img, 9, 75, 75)
    out = cv2.bitwise_and(cv_img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def add_frame(img, style="classic"):
    cv_img = to_bgr(img)
    h, w = cv_img.shape[:2]
    b = int(min(h, w) * 0.06)
    if style == "double":
        inner = cv2.copyMakeBorder(cv_img, b//2, b//2, b//2, b//2,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return cv2.copyMakeBorder(inner, b, b, b, b,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cv2.copyMakeBorder(cv_img, b, b, b, b, cv2.BORDER_CONSTANT,
                              value=FRAME_COLORS.get(style, (255, 255, 255)))


def make_favicon(img):
    return to_pil(img).convert("RGB").resize((64, 64), Image.LANCZOS)


def upscale_image(img, scale=2):
    pil = to_pil(img).convert("RGB")
    w, h = pil.size
    s = min(scale, 4096 / max(w, h))
    return pil.resize((max(2, int(w * s)), max(2, int(h * s))), Image.LANCZOS)


def smooth_image(img, iters=6):
    cv_img = to_bgr(img)
    for _ in range(iters):
        cv_img = cv2.bilateralFilter(cv_img, 9, 60, 60)
    return cv_img


def color_quantize(img, k=8):
    cv_img = to_bgr(img)
    Z = cv_img.reshape(-1, 3).astype(np.float32)
    _, label, center = cv2.kmeans(Z, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        5, cv2.KMEANS_RANDOM_CENTERS)
    return np.uint8(center)[label.flatten()].reshape(cv_img.shape)


def ink_outline(img):
    gray = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 9, 9)


def make_3d(img):
    """Anaglyph pseudo-3D (red/cyan)."""
    cv_img = to_bgr(img)
    shift = max(3, cv_img.shape[1] // 120)
    left, right = np.roll(cv_img, shift, 1), np.roll(cv_img, -shift, 1)
    out = np.stack([left[:, :, 0], left[:, :, 1], right[:, :, 2]], axis=2)
    return cv2.convertScaleAbs(out, alpha=1.05)


def make_2d(img):
    """Flat 2D image: neat cutout pasted on a plain white background."""
    return cutout_on_color(img, (255, 255, 255))


def change_color(img, color):
    hue = HUES.get(color)
    if hue is None:
        return to_bgr(img)
    hsv = cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = hue / 2
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.4, 50, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def polygon_mask(size, sides, rot=-90):
    mask = Image.new("L", size, 0)
    cx, cy = size[0] // 2, size[1] // 2
    r = min(size) // 2 - 5
    pts = [(cx + r * np.cos(np.radians(rot + i * 360 / sides)),
            cy + r * np.sin(np.radians(rot + i * 360 / sides))) for i in range(sides)]
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    return mask


def shape_album(img):
    """Album of the image masked as: triangle, square, pentagon, hexagon, heptagon, octagon."""
    pil = to_pil(img).convert("RGBA")
    tiles = []
    for name, n in [("Triangle", 3), ("Square", 4), ("Pentagon", 5),
                    ("Hexagon", 6), ("Heptagon", 7), ("Octagon", 8)]:
        tile = Image.new("RGB", pil.size, (245, 245, 245))
        tile.paste(pil, (0, 0), polygon_mask(pil.size, n))
        tiles.append(tile)
    tw, th = pil.width // 2, pil.height // 2
    canvas = Image.new("RGB", (tw * 3, th * 2), (230, 230, 230))
    for i, t in enumerate(tiles):
        canvas.paste(t.resize((tw, th), Image.LANCZOS), ((i % 3) * tw, (i // 3) * th))
    return canvas


def add_text(img, text, color=(255, 255, 0), effect="shadow"):
    pil = to_pil(img).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    size = max(20, pil.height // 12)
    font = None
    for fp in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(fp, size)
            break
        except Exception:
            pass
    if font is None:
        try:
            font = ImageFont.load_default(size)
        except TypeError:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((pil.width - tw) // 2, pil.height - th - 30)
    if effect == "shadow_only":                      # "letters only shadow"
        draw.text((pos[0] + 4, pos[1] + 4), text, fill=color, font=font)
    elif effect == "3d":                             # "3d letters"
        for dx, dy in ((4, 4), (3, 3), (2, 2)):
            draw.text((pos[0] + dx, pos[1] + dy), text, fill=(0, 0, 0, 255), font=font)
        draw.text(pos, text, fill=color, font=font)
    else:                                            # normal text with shadow
        draw.text((pos[0] + 3, pos[1] + 3), text, fill=(0, 0, 0, 255), font=font)
        draw.text(pos, text, fill=color, font=font)
    return pil.convert("RGB")


def make_video(img, seconds=3):
    """Image -> animation. GIF is used because Vercel has no ffmpeg."""
    pil = to_pil(img).convert("RGB")
    n = max(10, int(seconds) * 10)
    frames = []
    for i in range(n):
        t = i / (n - 1)
        ang = 6 * np.sin(2 * np.pi * t)               # gentle swing
        scale = 1 + 0.12 * np.sin(np.pi * t)          # gentle zoom
        f = pil.rotate(ang, expand=True, fillcolor=(255, 255, 255))
        f = f.resize((max(2, int(pil.width * scale)), max(2, int(pil.height * scale))), Image.LANCZOS)
        frames.append(f.resize(pil.size, Image.LANCZOS))
    fname = f"video_{uuid.uuid4().hex}.gif"
    frames[0].save(os.path.join(RESULT, fname), save_all=True,
                   append_images=frames[1:], duration=max(40, 1000 // n), loop=0)
    return fname


# ======================================================================
# ML PIPELINE  (Steps 4-6: Training -> Evaluation -> Deployment)
# ======================================================================
def extract_features(img):
    """Feature Engineering: HOG texture + RGB color histogram."""
    gray = cv2.cvtColor(cv2.resize(to_bgr(img), (64, 128)), cv2.COLOR_BGR2GRAY)
    feat = cv2.HOGDescriptor().compute(gray).flatten()
    hist = cv2.calcHist([cv2.resize(to_bgr(img), (64, 64))], [0, 1, 2], None,
                        [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    return np.concatenate([feat, hist])


def collect_gallery_labeled():
    """Data Collection -> training set from gallery images named cat_* / dog_*."""
    X, y = [], []
    for f in os.listdir(GALLERY):
        if f.rsplit(".", 1)[-1].lower() not in ALLOWED:
            continue
        fl = f.lower()
        label = 1 if fl.startswith("dog") else (0 if fl.startswith("cat") else None)
        if label is None:
            continue
        X.append(extract_features(Image.open(os.path.join(GALLERY, f)).convert("RGB")))
        y.append(label)
    return np.array(X), np.array(y)


def run_classify(img):
    """Step 6: Model Deployment."""
    if not SKLEARN:
        return {"label": None, "error": "scikit-learn not installed."}
    if not os.path.exists(MODEL_PATH):
        return {"label": None,
                "error": "Model not trained yet. Save cat_* / dog_* images to the gallery, then press Train."}
    with open(MODEL_PATH, "rb") as f:
        b = pickle.load(f)
    feat = b["scaler"].transform(extract_features(img).reshape(1, -1))
    proba = b["model"].predict_proba(feat)[0]
    return {"label": ["cat", "dog"][int(b["model"].predict(feat)[0])],
            "confidence": round(float(proba.max()) * 100, 1)}


# ======================================================================
# PROMPT PARSER
# ======================================================================
def detect_color(prompt):
    for c in sorted(COLORS, key=len, reverse=True):     # "light blue" before "blue"
        if c in prompt:
            return c
    return None


def parse_prompt(prompt, user_text=""):
    p = prompt.lower().strip()
    found = set()
    for name, kws in KEYWORDS.items():
        if any(k in p for k in kws):
            found.add(name)

    style = "classic"                                   # frame style keywords
    for s, kws in FRAME_STYLES.items():
        if any(k in p for k in kws):
            style = s

    m = re.search(r"(\d+)\s*(?:seconds?|secs?|s)\b", p)  # "video 5 seconds"
    seconds = int(m.group(1)) if m else 3

    color = detect_color(p)

    text = (user_text or "").strip()                    # overlay text
    if not text:
        m = re.search(r"text\s*[:\-]\s*(.+)", p)
        if m:
            text = m.group(1).strip()

    effect = "shadow"
    if re.search(r"shadow\s*only|only\s*shadow|letters?\s+only", p):
        effect = "shadow_only"
    elif "text" in found and re.search(r"3-?d", p):
        effect = "3d"
    # "3d text" should not trigger the anaglyph effect unless "3d image/effect" is written
    if "three_d" in found and not re.search(r"3-?d\s+(image|effect)|anaglyph", p):
        found.discard("three_d")

    tasks = [t for t in ORDER if t in found]
    return tasks, style, color, text, effect, seconds


def apply_operations(img, tasks, style, color, text, effect, seconds, bg=None):
    result, video = img, None
    for t in tasks:
        if   t == "remove_background": result = remove_background(result)
        elif t == "cut_paste":
            result = paste_on_image(result, bg) if bg is not None else cutout_on_color(result, (255, 255, 255))
        elif t == "white_background":  result = cutout_on_color(result, (255, 255, 255))
        elif t == "bg_color":          result = cutout_on_color(result, COLORS.get(color or "white"))
        elif t == "black_white":       result = to_black_white(result)
        elif t == "cartoon":           result = cartoon_effect(result)
        elif t == "ghibli":            result = ghibli_effect(result)
        elif t == "smooth":            result = smooth_image(result)
        elif t == "color_quantize":    result = color_quantize(result)
        elif t == "change_color":
            if color: result = change_color(result, color)
        elif t == "two_d":             result = make_2d(result)
        elif t == "ink_outline":       result = ink_outline(result)
        elif t == "three_d":           result = make_3d(result)
        elif t == "shape_album":       result = shape_album(result)
        elif t == "frame":             result = add_frame(result, style)
        elif t == "text":
            if text: result = add_text(result, text, COLORS.get(color or "yellow"), effect)
        elif t == "favicon":           result = make_favicon(result)
        elif t == "upscale":           result = upscale_image(result)
        elif t == "video":             video = make_video(result, seconds)
    return result, video


# ======================================================================
# ROUTES
# ======================================================================
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/index.css")
def css():
    return send_from_directory(".", "index.css", mimetype="text/css")


@app.route("/keywords")
def keywords():
    return jsonify({"features": KEYWORDS, "frames": list(FRAME_STYLES), "colors": list(COLORS)})


# ---------- Step 1: Data Collection ----------
@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("image")
    if not f or not allowed_file(f.filename or ""):
        return jsonify({"success": False, "error": "Please upload png/jpg/jpeg/bmp/webp"}), 400
    fid = uuid.uuid4().hex
    f.save(os.path.join(UPLOAD, f"{fid}.{f.filename.rsplit('.', 1)[1].lower()}"))
    return jsonify({"success": True, "file_id": fid})


# ---------- Steps 2-6: Process ----------
@app.route("/process", methods=["POST"])
def process():
    data = request.get_json() or {}
    img = load_image(data.get("file_id", ""))
    if img is None:
        return jsonify({"success": False, "error": "Image not found. Upload an image first."}), 400

    bg = load_image(data.get("bg_file_id", "")) if data.get("bg_file_id") else None
    tasks, style, color, text, effect, seconds = parse_prompt(data.get("prompt", ""),
                                                              data.get("user_text", ""))
    picker = (data.get("bg_color") or "").strip().lower()
    if picker in COLORS:
        if ("bg_color" in tasks or "change_color" in tasks) and not color:
            color = picker
        if "text" in tasks and not color:
            color = picker

    notes = []
    if "change_color" in tasks and not color:
        notes.append("No color chosen — 'change color' was skipped.")
    if "bg_color" in tasks and not color:
        notes.append("No color chosen — white background used.")

    classify_result = None
    if "classify" in tasks:
        tasks.remove("classify")
        classify_result = run_classify(img)
        if not tasks:
            return jsonify({"success": True, "tasks": ["classify"],
                            "classify": classify_result, "notes": notes})

    if not tasks:
        return jsonify({"success": False,
                        "error": "No known keywords found. Try: remove background, cartoon, "
                                 "ghibli, frame, upscale, text: Hello ..."}), 400

    result, video = apply_operations(img, tasks, style, color, text, effect, seconds, bg)
    if isinstance(result, np.ndarray):
        result = to_pil(result)
    result = result.convert("RGB")

    fname = f"result_{uuid.uuid4().hex}.png"
    result.save(os.path.join(RESULT, fname))
    buf = io.BytesIO()
    result.save(buf, "PNG")

    return jsonify({"success": True, "tasks": tasks,
                    "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                    "filename": fname, "download": f"/download/{fname}",
                    "video": f"/download/{video}" if video else None,
                    "classify": classify_result, "notes": notes, "rembg": REMBG})


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(RESULT, filename, as_attachment=True)


# ---------- Gallery database (server) ----------
@app.route("/gallery")
def gallery_list():
    conn = get_db()
    rows = conn.execute("SELECT id,name,prompt,created FROM gallery ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify({"success": True,
                    "items": [{"id": r[0], "name": r[1], "prompt": r[2], "created": r[3],
                               "url": f"/gallery/image/{r[0]}"} for r in rows]})


@app.route("/gallery/image/<int:gid>")
def gallery_image(gid):
    conn = get_db()
    row = conn.execute("SELECT filename FROM gallery WHERE id=?", (gid,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(GALLERY, row[0])


@app.route("/gallery/save", methods=["POST"])
def gallery_save():
    data = request.get_json() or {}
    img = load_image(data.get("file_id", ""))
    if img is None:
        return jsonify({"success": False, "error": "Image not found"}), 404
    name = re.sub(r"[^A-Za-z0-9_\-]", "_", (data.get("name") or f"image_{uuid.uuid4().hex[:6]}").strip())
    fname = f"{uuid.uuid4().hex[:8]}_{name}.png"
    img.convert("RGB").save(os.path.join(GALLERY, fname))
    conn = get_db()
    cur = conn.execute("INSERT INTO gallery(filename,name,prompt) VALUES(?,?,?)",
                       (fname, name, data.get("prompt", "")))
    conn.commit()
    gid = cur.lastrowid
    conn.close()
    return jsonify({"success": True, "id": gid})


@app.route("/gallery/delete/<int:gid>", methods=["POST"])
def gallery_delete(gid):
    conn = get_db()
    row = conn.execute("SELECT filename FROM gallery WHERE id=?", (gid,)).fetchone()
    if row:
        conn.execute("DELETE FROM gallery WHERE id=?", (gid,))
        conn.commit()
        try:
            os.remove(os.path.join(GALLERY, row[0]))
        except OSError:
            pass
    conn.close()
    return jsonify({"success": True})


# ---------- Steps 4+5: Training + Evaluation ----------
@app.route("/train", methods=["POST"])
def train():
    if not SKLEARN:
        return jsonify({"success": False, "error": "scikit-learn is not installed."}), 400
    X, y = collect_gallery_labeled()
    if len(X) < 6 or len(set(y.tolist())) < 2:
        return jsonify({"success": False,
                        "error": "Need at least 6 labeled gallery images "
                                 "(names starting with cat_... and dog_...)."}), 400
    strat = y if min(np.bincount(y)) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)
    pred = model.predict(scaler.transform(Xte))
    acc = accuracy_score(yte, pred)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    return jsonify({"success": True, "accuracy": round(acc * 100, 1), "images_used": len(X),
                    "report": classification_report(yte, pred, target_names=["cat", "dog"],
                                                    zero_division=0)})


# ======================================================================
handler = app                       # Vercel entry point
if __name__ == "__main__":
    app.run(debug=True, port=5000)