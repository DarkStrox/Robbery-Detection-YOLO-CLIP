import os

# Base paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP_DIR = os.path.join(PROJECT_DIR, "openAI clip")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

# Existing local Florence-2 models cache directory
FLORENCE_MODELS_DIR = r"E:\Aitronix Projects\Robber Detection\models"

os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Point HuggingFace & Torch caches to existing local models folder
os.environ["HF_HOME"] = FLORENCE_MODELS_DIR
os.environ["TORCH_HOME"] = FLORENCE_MODELS_DIR
os.environ["TRANSFORMERS_CACHE"] = FLORENCE_MODELS_DIR

# YOLO Model Settings
YOLO_GUNS_WEIGHTS = r"E:\Aitronix Projects\Robber Detection\Yolo Guns Repo\Weapons-and-Knives-Detector-with-YOLOv8\runs\detect\Normal_Compressed\weights\best.pt"
YOLO_CONF_THRESHOLD = 0.45  # Filter out low-confidence monitor false positives

# Trigger Labels to Fire Stage 2 & Stage 3
TRIGGER_CLASSES = ["gun", "knife", "weapon", "pistol", "handgun", "rifle", "balaclava", "mask"]

# Crop & Enhancement Settings
CROP_BUFFER_PCT = 0.85  # 85% padding buffer around target to include person & scene context
ENHANCE_UPSCALE_FACTOR = 2.0
SHARPEN_IMAGE = True

# CLIP Classification Settings
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_CATEGORIES = [
    "A photo of a normal peaceful scene or store transaction with a counter",
    "A photo of an armed robbery in progress with a weapon",
    "A photo of a suspect brandishing a gun or knife",
    "A photo of a masked person committing a crime or assault",
    "A photo of vandalism or violent physical confrontation"
]

# Florence-2 VLM Verification Keywords
VLM_THREAT_KEYWORDS = [
    "gun", "weapon", "pistol", "handgun", "rifle", "knife", 
    "balaclava", "mask", "pointing", "armed", "threat", "robbery", 
    "holding a gun", "aiming", "shotgun", "firearm", "assault"
]

# Florence-2 VLM Settings
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"
FLORENCE_REASONING_PROMPT = "<DETAILED_CAPTION>"


