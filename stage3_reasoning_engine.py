import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, List

# Monkey-patch check_imports for Windows & HuggingFace dynamic code loading
import transformers.dynamic_module_utils
orig_check_imports = transformers.dynamic_module_utils.check_imports
def custom_check_imports(filename):
    try:
        imports = orig_check_imports(filename)
        if "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports
    except ImportError as e:
        if "flash_attn" in str(e):
            return []
        raise e
transformers.dynamic_module_utils.check_imports = custom_check_imports

from transformers import AutoProcessor, AutoModelForCausalLM, CLIPProcessor, CLIPModel

from config import (
    CLIP_DIR,
    CLIP_MODEL_ID,
    CLIP_CATEGORIES,
    FLORENCE_MODELS_DIR,
    FLORENCE_MODEL_ID,
    FLORENCE_REASONING_PROMPT
)

class CLIPClassifier:
    """
    Zero-shot categorization using openai/clip-vit-base-patch32 stored locally in 'openAI clip'.
    """
    def __init__(self, clip_dir: str = CLIP_DIR):
        print(f"[Stage 3A] Initializing OpenAI CLIP Classifier from: {clip_dir}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load local or downloaded CLIP model
        if os.path.exists(clip_dir) and any(f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".pt") for f in os.listdir(clip_dir)):
            print(f"[Stage 3A] Loading local OpenAI CLIP weights from {clip_dir}")
            self.model = CLIPModel.from_pretrained(clip_dir, local_files_only=True).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(clip_dir, local_files_only=True)
        else:
            print(f"[Stage 3A] Loading OpenAI CLIP ({CLIP_MODEL_ID})...")
            self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID, cache_dir=clip_dir).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=clip_dir)

    def classify_crop(self, crop_image: Image.Image, categories: List[str] = CLIP_CATEGORIES) -> Tuple[str, float, Dict[str, float]]:
        """
        Calculates zero-shot category probabilities for the crop.
        """
        if crop_image is None:
            return "Unknown", 0.0, {}

        inputs = self.processor(text=categories, images=crop_image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image # image-to-text similarity
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

        scores_dict = {}
        for cat, prob in zip(categories, probs):
            clean_label = cat.replace("A photo of a ", "").replace("A photo of an ", "")
            scores_dict[clean_label] = round(float(prob) * 100, 2)

        top_idx = int(np.argmax(probs))
        top_category = list(scores_dict.keys())[top_idx]
        top_score = list(scores_dict.values())[top_idx]

        return top_category, top_score, scores_dict


class FlorenceVLMReasoningEngine:
    """
    Florence-2 Vision-Language Model for generating natural language explanations of threat scenes.
    Uses local files directly from FLORENCE_MODELS_DIR (no online re-downloading).
    """
    def __init__(self, models_dir: str = FLORENCE_MODELS_DIR, model_id: str = FLORENCE_MODEL_ID):
        print(f"[Stage 3B] Initializing Florence-2 VLM Reasoning Engine from local cache: {models_dir}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["HF_HOME"] = models_dir

        try:
            # Try loading 100% offline from local cache first
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=models_dir,
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=models_dir,
                local_files_only=True
            ).to(self.device)
            print("[Stage 3B] Florence-2 VLM loaded successfully from LOCAL cache (Offline mode)!")
        except Exception as e:
            print(f"[Stage 3B WARNING] Offline load failed ({e}), trying standard cache load...")
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=models_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=models_dir
            ).to(self.device)

    def generate_reasoning(
        self, 
        crop_image: Image.Image, 
        task_prompt: str = FLORENCE_REASONING_PROMPT
    ) -> Tuple[str, float]:
        """
        Generates explanatory reasoning sentence for the cropped target region.
        """
        if crop_image is None:
            return "No image provided for reasoning.", 0.0

        start_time = time.time()
        inputs = self.processor(text=task_prompt, images=crop_image, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=150,
                num_beams=1
            )

        elapsed_ms = (time.time() - start_time) * 1000
        raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_result = self.processor.post_process_generation(
            raw_text,
            task=task_prompt.split(" ")[0],
            image_size=(crop_image.width, crop_image.height)
        )

        description = parsed_result.get(task_prompt.split(" ")[0], "")
        if isinstance(description, dict):
            description = str(description)

        reasoning_sentence = str(description).strip()
        return reasoning_sentence, round(elapsed_ms, 2)


class ReasoningEngine:
    """
    Combined Stage 3 Reasoning Engine combining CLIP and Florence-2 VLM.
    """
    def __init__(self, clip_dir: str = CLIP_DIR, florence_dir: str = FLORENCE_MODELS_DIR):
        self.clip = CLIPClassifier(clip_dir=clip_dir)
        self.vlm = None  # Lazy load Florence-2 VLM on demand

    def init_vlm(self):
        if self.vlm is None:
            self.vlm = FlorenceVLMReasoningEngine(models_dir=FLORENCE_MODELS_DIR)

    def analyze_threat_crop(self, crop_image: Image.Image) -> Dict[str, Any]:
        """
        Full Stage 3 analysis: CLIP zero-shot classification + Florence-2 VLM reasoning generation.
        """
        if crop_image is None:
            return {}

        top_cat, top_score, clip_scores = self.clip.classify_crop(crop_image)
        self.init_vlm()
        reasoning_sentence, inference_ms = self.vlm.generate_reasoning(crop_image)

        return {
            "top_category": top_cat,
            "top_score": top_score,
            "clip_scores": clip_scores,
            "reasoning_sentence": reasoning_sentence,
            "vlm_latency_ms": inference_ms
        }

if __name__ == "__main__":
    print("Testing Stage 3 Reasoning Engine...")
    engine = ReasoningEngine()
    dummy = Image.new("RGB", (224, 224), color=(100, 50, 50))
    res = engine.analyze_threat_crop(dummy)
    print(f"Stage 3 Test Result: {res}")
