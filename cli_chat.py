"""
mental_health_chatbot/src/cli_chat.py

Command-line interface for the Mental Health Chatbot.
Usage: python src/cli_chat.py --models_dir ./models
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chatbot_engine import MentalHealthChatbot


BANNER = """
╔══════════════════════════════════════════════════════════╗
║         🧠  Mental Health Support Chatbot  🧠            ║
║                                                          ║
║  This chatbot offers empathetic, AI-powered support.    ║
║  It is NOT a substitute for professional mental health  ║
║  care. In an emergency, call 988 (US) or your local     ║
║  crisis line.                                            ║
║                                                          ║
║  🔍 XAI explanations are shown with every response      ║
║     to provide transparency about emotion detection.    ║
║                                                          ║
║  Commands:  quit / exit → end session                   ║
║             clear       → clear conversation            ║
╚══════════════════════════════════════════════════════════╝
"""


def format_emotion_bar(label: str, score: float, width: int = 20) -> str:
    filled = int(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"  {label:12s} [{bar}] {score:.1%}"


def run_cli(models_dir: str):
    print(BANNER)
    chatbot = MentalHealthChatbot(models_dir=models_dir)

    print("\nType your message below. Take your time.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nTake care. Goodbye. 💙")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "exit", "bye"):
            print("\nTake care. Goodbye. 💙")
            break
        elif cmd == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)
            print("\nConversation cleared.\n")
            continue

        # XAI is always enabled
        result = chatbot.chat(user_input, explain=True)

        # Print response
        print(f"\n  Bot: {result['response']}\n")

        # Emotion analysis
        emotion = result["emotion"]
        conf    = result["confidence"]
        risk    = result["risk_level"].upper()
        print(f"  ┌── Analysis ───────────────────────────────────┐")
        print(f"  │  Emotion   : {emotion.upper():10s}  ({conf:.1%} confidence)")
        print(f"  │  Risk Level: {risk}")
        print(f"  │")
        print(f"  │  Score Breakdown:")
        for label, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
            print(f"  │  {format_emotion_bar(label, score)}")
        print(f"  └───────────────────────────────────────────────┘")

        # XAI explanation - simplified, always shown
        if result.get('xai_explanation'):
            print(f"\n  🔍 {result['xai_explanation']}")

        # Helplines
        if result.get("show_helpline") and result.get("helplines"):
            print(f"\n  ┌── Support Resources ──────────────────────────┐")
            for h in result["helplines"]:
                print(f"  │  • {h['name']}")
                print(f"  │    {h['number']}")
                print(f"  │    {h['note']}")
            print(f"  └───────────────────────────────────────────────┘")

        print()  # spacing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mental Health Chatbot CLI")
    parser.add_argument(
        "--models_dir", type=str, default="models",
        help="Path to the models directory (default: ./models)"
    )
    args = parser.parse_args()
    run_cli(args.models_dir)
