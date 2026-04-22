"""
mental_health_chatbot/src/chatbot_engine.py

Core inference engine for the Mental Health Chatbot.
Handles: Emotion classification, Risk assessment, DialoGPT response,
         XAI explanation (SHAP), Helpline suggestions.
"""

import json
import os
import re
import torch
import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import shap
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    pipeline
)


# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION & HELPLINES
# ─────────────────────────────────────────────────────────────

HELPLINES = {
    "crisis": [
        {"name": "iCall (TISS)",
         "number": "9152987821",
         "note": "Free, confidential counselling by trained psychologists — Mon–Sat, 8AM–10PM"},
        {"name": "Vandrevala Foundation Helpline",
         "number": "1860-2662-345",
         "note": "24/7 mental health support in English and Hindi"},
        {"name": "AASRA",
         "number": "9820466627",
         "note": "24/7 crisis intervention and suicide prevention helpline"},
    ],
    "anxiety": [
        {"name": "iCall (TISS)",
         "number": "9152987821",
         "note": "Talk to a trained counsellor — Mon–Sat, 8AM–10PM"},
        {"name": "Vandrevala Foundation Helpline",
         "number": "1860-2662-345",
         "note": "24/7 support for anxiety, stress and mental health concerns"},
        {"name": "The MINDS Foundation",
         "number": "https://mindsfoundation.org",
         "note": "Mental health awareness and support resources across India"},
    ],
    "depression": [
        {"name": "iCall (TISS)",
         "number": "9152987821",
         "note": "Free psychological counselling — Mon–Sat, 8AM–10PM"},
        {"name": "Snehi",
         "number": "044-24640050",
         "note": "Emotional support helpline available daily, 8AM–10PM"},
        {"name": "Vandrevala Foundation Helpline",
         "number": "1860-2662-345",
         "note": "24/7 confidential support for depression and low mood"},
    ],
    "grief": [
        {"name": "iCall (TISS)",
         "number": "9152987821",
         "note": "Speak with a counsellor about loss and grief — Mon–Sat, 8AM–10PM"},
        {"name": "Snehi",
         "number": "044-24640050",
         "note": "Compassionate emotional support during difficult times"},
        {"name": "YourDOST",
         "number": "https://yourdost.com",
         "note": "Online counselling and emotional wellness platform"},
    ],
    "general": [
        {"name": "iCall (TISS)",
         "number": "9152987821",
         "note": "Free, professional mental health support — Mon–Sat, 8AM–10PM"},
        {"name": "YourDOST",
         "number": "https://yourdost.com",
         "note": "Online counselling platform with licensed Indian therapists"},
        {"name": "The MINDS Foundation",
         "number": "https://mindsfoundation.org",
         "note": "Mental health resources, community support and awareness"},
    ]
}

EMPATHY_STARTERS = {
    "depression":  "I hear how heavy things feel for you right now.",
    "anxiety":     "It sounds like you're carrying a lot of worry.",
    "crisis":      "I'm really glad you're reaching out — that takes courage.",
    "anger":       "It's completely valid to feel frustrated by that.",
    "grief":       "I'm so sorry for what you're going through.",
    "stress":      "That sounds genuinely overwhelming.",
    "neutral":     "Thank you for sharing that with me."
}


# ─────────────────────────────────────────────────────────────
# 2. EMOTION & RISK CLASSIFIER
# ─────────────────────────────────────────────────────────────

class EmotionRiskClassifier:
    """
    DistilBERT-based emotion classifier + rule-based risk assessment.
    """

    RISK_KEYWORDS = {
        "high": [
            "suicide", "kill myself", "end my life", "don't want to live",
            "dont want to live", "self harm", "cut myself", "hurt myself",
            "overdose", "no reason to live", "better off dead"
        ],
        "medium": [
            "hopeless", "worthless", "can't go on", "cant go on",
            "give up on life", "no point in living", "disappear forever"
        ],
        "low": [
            "really sad", "depressed", "very anxious", "stressed out",
            "completely lost", "alone"
        ]
    }

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        print(f"  Loading emotion classifier from: {model_path}")

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path
        ).to(device)
        self.model.eval()

        # Load label mappings
        config_path = Path(model_path) / "label_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.id2label = {int(k): v for k, v in config["id2label"].items()}
        else:
            # Fallback defaults if config missing
            self.id2label = {
                0: "depression", 1: "anxiety", 2: "crisis",
                3: "anger", 4: "grief", 5: "stress", 6: "neutral"
            }

        print(f"  Emotion labels: {list(self.id2label.values())}")

    def classify_emotion(self, text: str) -> Dict:
        """Return predicted emotion and per-class probabilities."""
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=128, padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        pred_id = int(np.argmax(probs))
        pred_label = self.id2label[pred_id]

        return {
            "emotion": pred_label,
            "confidence": float(probs[pred_id]),
            "all_scores": {self.id2label[i]: float(p) for i, p in enumerate(probs)}
        }

    def assess_risk(self, text: str) -> str:
        """Rule-based risk level: high / medium / low / none."""
        text_lower = text.lower()
        for level in ("high", "medium", "low"):
            if any(kw in text_lower for kw in self.RISK_KEYWORDS[level]):
                return level
        return "none"


# ─────────────────────────────────────────────────────────────
# 3. XAI EXPLAINER (SHAP)
# ─────────────────────────────────────────────────────────────

class XAIExplainer:
    """
    Uses SHAP to explain *why* a specific emotion was predicted.
    Uses a fast TextClassificationPipeline under the hood.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
        # Load explicitly to avoid unsupported pipeline kwargs on older transformers
        _tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        _model = DistilBertForSequenceClassification.from_pretrained(model_path, )
        self.pipe = pipeline(
            "text-classification",
            model=_model,
            tokenizer=_tokenizer,
            device=0 if device == "cuda" else -1,
            top_k=None,
            truncation=True,
            max_length=128,
        )

        # SHAP explainer wraps the pipeline
        self.explainer = shap.Explainer(self.pipe)
        print("  XAI (SHAP) explainer ready.")

    def explain(self, text: str, emotion: str, top_n: int = 5) -> Dict:
        """
        Return the top words that pushed toward the given emotion prediction.
        """
        try:
            shap_values = self.explainer([text])

            # Find index of target emotion label
            output_names = shap_values.output_names
            if emotion not in output_names:
                return {"top_words": [], "explanation": "Explanation unavailable."}

            emotion_idx = output_names.index(emotion)
            sv = shap_values[0, :, emotion_idx]

            # Pair tokens with SHAP values
            tokens = shap_values.data[0]
            token_shap = list(zip(tokens, sv.values))

            # Sort by absolute contribution (descending)
            token_shap.sort(key=lambda x: abs(x[1]), reverse=True)

            # Filter out punctuation-only tokens
            top = [
                {"word": t, "impact": round(float(v), 4)}
                for t, v in token_shap[:top_n]
                if re.search(r'\w', t)
            ]

            explanation = f"The model detected **{emotion}** mainly because of: "
            explanation += ", ".join([f"'{w['word']}'" for w in top[:3]])
            explanation += "."

            return {"top_words": top, "explanation": explanation}

        except Exception as e:
            return {
                "top_words": [],
                "explanation": f"XAI explanation unavailable: {str(e)}"
            }


# ─────────────────────────────────────────────────────────────
# 4. DIALOGPT RESPONSE GENERATOR
# ─────────────────────────────────────────────────────────────

from transformers import LogitsProcessor, LogitsProcessorList

class SafeLogitsProcessor(LogitsProcessor):
    """Clamp logits to prevent NaN/Inf errors during multinomial sampling."""
    def __call__(self, input_ids, scores):
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        scores = torch.clamp(scores, min=-1e4, max=1e4)
        return scores


class DialogGPTResponder:
    """
    Generates empathetic responses using fine-tuned DialoGPT-medium.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        print(f"  Loading DialoGPT from: {model_path}")
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        import os
    
    # Convert to absolute path and normalize
        model_path = os.path.abspath(model_path)
    
    # Load tokenizer directly from file paths (bypass from_pretrained validation)
        vocab_file = os.path.join(model_path, 'vocab.json')
        merges_file = os.path.join(model_path, 'merges.txt')
    
        self.tokenizer = GPT2Tokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token='<|endoftext|>',
            bos_token='<|endoftext|>',
            eos_token='<|endoftext|>'
        )
    
    # Load model using GPT2LMHeadModel instead of Auto
        self.model = GPT2LMHeadModel.from_pretrained(
        model_path
        ).to(device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
    
        self.model.eval()
        self.device = device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("  DialoGPT ready.")

    # Template responses for each emotion 
    RESPONSE_TEMPLATES = {
        "depression": [
            "I hear how heavy things feel right now. Depression can make everything seem impossible, but you don't have to face this alone. Small steps matter — even reaching out like this takes strength.",
            "It sounds like you're carrying a lot of pain. When we're depressed, it can feel like there's no way forward, but that feeling isn't the truth. You deserve support and care.",
            "I'm really glad you're talking about this. Depression lies to us — it tells us things won't get better, but they can. What you're feeling is real, and so is the possibility of healing.",
            "Living with depression is exhausting in a way that's hard for others to understand. Your feelings are completely valid. Please know that help is available and you don't have to go through this alone.",
        ],
        "anxiety": [
            "It sounds like you're carrying a lot of worry right now. Anxiety can feel overwhelming, but you're not alone in this. Taking things one moment at a time can help when everything feels too much.",
            "I can hear how much tension you're holding. When anxiety takes over, it can feel like the worry will never stop — but it can ease with the right support. You're taking a good step by talking about it.",
            "Anxiety has a way of making everything feel urgent and threatening. Your feelings are real and valid. Let's slow down together — you're safe in this moment.",
            "Feeling this anxious is really hard. It takes courage to acknowledge it. Remember that anxiety, as overwhelming as it feels, doesn't have to control your life.",
        ],
        "crisis": [
            "I'm really glad you reached out — that took courage. What you're feeling right now is serious and you deserve immediate support. Please contact a crisis line — trained counselors are available 24/7.",
            "I'm very concerned about what you've shared. Your life has value, and you don't have to face this alone. Please reach out to a crisis helpline right now — they truly want to help.",
        ],
        "anger": [
            "It's completely understandable to feel angry — your emotions are valid. Anger often comes from pain or feeling unheard. I'm here to listen without judgment.",
            "That frustration makes a lot of sense given what you're going through. Anger is a signal that something important to you has been affected. Let's talk through it.",
            "I hear how frustrated you are, and that's okay. Sometimes anger is the only way we know how to express deep hurt. You're safe to express how you feel here.",
        ],
        "grief": [
            "I'm so sorry for what you're going through. Grief is one of the hardest things a person can experience, and there's no right or wrong way to feel it. I'm here with you.",
            "Losing someone or something you love leaves a wound that takes time. Your grief is a reflection of how deeply you cared. Please be gentle with yourself.",
            "Grief can feel like waves — sometimes calm, sometimes overwhelming. Whatever you're feeling right now is okay. You don't have to rush through this.",
        ],
        "stress": [
            "That sounds genuinely overwhelming. When stress piles up, it can feel like there's no breathing room. Let's slow down — you don't have to solve everything at once.",
            "I hear you — you're dealing with a lot right now. Stress at that level is exhausting. Breaking things into smaller pieces can help, but first, just know that it's okay to feel this way.",
            "It makes complete sense that you're stressed given everything you're facing. You're not weak for feeling this way — you're human. Is there one thing we can focus on together?",
        ],
        "neutral": [
            "Thank you for sharing that with me. I'm here to listen and support you however I can. Please feel free to share more about what's on your mind.",
            "I appreciate you opening up. Whatever you're going through, you don't have to face it alone. I'm here to talk.",
            "I'm glad you reached out. Sometimes just putting our thoughts into words can help. Tell me more about what's been going on for you.",
        ],
    }

    def generate(
        self,
        user_text: str,
        emotion: str,
        **kwargs,
    ) -> str:
        """Generate an empathetic response using curated templates (instant, no gibberish)."""
        import random
        if emotion not in self.RESPONSE_TEMPLATES:
            emotion = "neutral"
        templates = self.RESPONSE_TEMPLATES[emotion]
        return random.choice(templates)




# ─────────────────────────────────────────────────────────────
# 5. DECISION LAYER (RULES ENGINE)
# ─────────────────────────────────────────────────────────────

class DecisionLayer:
    """
    Orchestrates the pipeline: classifier → response → XAI → helplines.
    """

    CRISIS_OVERRIDE_RESPONSE = (
        "I'm very concerned about what you've shared. "
        "Your life has value, and you don't have to face this alone. "
        "Please reach out to a crisis line right now — trained counselors are "
        "available 24/7 and want to help you through this."
    )

    def decide(
        self,
        emotion: str,
        risk_level: str,
        raw_response: str
    ) -> Tuple[str, bool, str]:
        """
        Returns: (final_response, show_helpline, helpline_category)
        """
        show_helpline = False
        helpline_cat  = "general"

        if risk_level == "high" or emotion == "crisis":
            return self.CRISIS_OVERRIDE_RESPONSE, True, "crisis"

        if risk_level == "medium":
            show_helpline = True
            helpline_cat  = emotion if emotion in HELPLINES else "general"

        elif emotion in ("depression", "anxiety", "grief") and risk_level != "none":
            show_helpline = True
            helpline_cat  = emotion

        return raw_response, show_helpline, helpline_cat


# ─────────────────────────────────────────────────────────────
# 6. MAIN CHATBOT CLASS
# ─────────────────────────────────────────────────────────────

class MentalHealthChatbot:
    """
    Full pipeline:
      User Text
         ↓ EmotionRiskClassifier (DistilBERT)
         ↓ DecisionLayer (Rules)
         ↓ DialogGPTResponder (Fine-tuned)
         ↓ XAIExplainer (SHAP)
         ↓ Safe Output + Helplines
    """

    def __init__(self, models_dir: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"\n{'='*55}")
        print("  Loading Mental Health Chatbot")
        print(f"  Device: {device.upper()}")
        print(f"{'='*55}")

        emotion_model_path = "GaneshReddy69/emotion-classifier"
        dialogpt_path      = os.path.abspath(os.path.join(models_dir, "dialogpt_finetuned", "final_model"))

        self.emotion_clf = EmotionRiskClassifier(emotion_model_path, device)
        self.dialogpt    = DialogGPTResponder(dialogpt_path, device)
        self.xai         = XAIExplainer(emotion_model_path, device)
        self.decision    = DecisionLayer()

        print(f"\nChatbot ready!\n{'='*55}\n")

    def chat(self, user_input: str, explain: bool = True) -> Dict:
        """
        Process user input through full pipeline.
        
        XAI is ALWAYS enabled by default to provide transparency.
        
        Returns a dict with:
          - response       : str
          - emotion        : str
          - confidence     : float
          - risk_level     : str
          - xai_explanation: str
          - xai_top_words  : list
          - helplines      : list (empty if not triggered)
          - show_helpline  : bool
        """
        user_input = user_input.strip()
        if not user_input:
            return {"response": "Please share what's on your mind.", "emotion": "neutral"}

        # Step 1: Classify emotion
        emotion_result = self.emotion_clf.classify_emotion(user_input)
        emotion     = emotion_result["emotion"]
        confidence  = emotion_result["confidence"]
        all_scores  = emotion_result["all_scores"]

        # Step 2: Risk assessment
        risk_level = self.emotion_clf.assess_risk(user_input)

        # Step 3: Generate raw response
        try:
            raw_response = self.dialogpt.generate(user_input, emotion)
        except RuntimeError as e:
            print(f"[WARN] DialoGPT generation failed, using fallback: {e}")
            raw_response = EMPATHY_STARTERS.get(emotion, EMPATHY_STARTERS["neutral"])

        # Step 4: Decision layer
        final_response, show_helpline, helpline_cat = self.decision.decide(
            emotion, risk_level, raw_response
        )

        # Step 5: XAI explanation
        xai_result = self.xai.explain(user_input, emotion) if explain else {
            "explanation": "", "top_words": []
        }

        # Step 6: Helplines
        helplines = HELPLINES.get(helpline_cat, []) if show_helpline else []

        return {
            "response":        final_response,
            "emotion":         emotion,
            "confidence":      round(confidence, 3),
            "risk_level":      risk_level,
            "all_scores":      all_scores,
            "xai_explanation": xai_result["explanation"],
            "xai_top_words":   xai_result["top_words"],
            "helplines":       helplines,
            "show_helpline":   show_helpline
        }

    def pretty_print(self, result: Dict, user_input: str):
        """Nicely print a chatbot result to the console with XAI always shown."""
        print(f"\n{'═'*70}")
        print(f" YOU:  {user_input}")
        print(f"{'═'*70}")
        print(f" BOT:  {result['response']}")
        
        # print(f"\n{'─'*70}")
        # print(f"EMOTION ANALYSIS")
        # print(f"{'─'*70}")
        # print(f"  Detected: {result['emotion'].upper()} ({result['confidence']:.1%} confidence)")
        # print(f"  Risk Level: {result['risk_level'].upper()}")

        # Simplified XAI - just the explanation text
        if result.get('xai_explanation'):
            print(f"\n  🔍 {result['xai_explanation']}")

        if result.get("helplines"):
            print(f"\n{'─'*70}")
            print(f"SUPPORT RESOURCES")
            print(f"{'─'*70}")
            for h in result["helplines"]:
                print(f"  • {h['name']}: {h['number']}")
                print(f"    {h['note']}")

        print(f"{'═'*70}\n")
