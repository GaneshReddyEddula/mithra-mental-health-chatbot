

import os
import sys
import json
import torch
import gradio as gr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chatbot_engine import MentalHealthChatbot

MODELS_DIR = r"C:\Users\Ganesh\OneDrive\Desktop\mental health chatbot\Local\models"
print(f"Loading chatbot from: {MODELS_DIR}")
chatbot = MentalHealthChatbot(models_dir=MODELS_DIR)

def respond(user_message, history):
    """Main response handler for Gradio."""
    if not user_message.strip():
        return history, ""

    result = chatbot.chat(user_message, explain=True)

    history = history or []

    
    bot_response = result["response"]
    if result.get('xai_explanation'):
        bot_response += f"\n\n *{result['xai_explanation']}*"

    history.append((user_message, bot_response))

    
    helplines_panel = ""
    if result.get("show_helpline") and result.get("helplines"):
        helplines_panel = "## Support Resources\n\n"
        for h in result["helplines"]:
            helplines_panel += f"### {h['name']}\n"
            helplines_panel += f"**{h['number']}**\n\n"
            helplines_panel += f"_{h['note']}_\n\n"
    else:
        helplines_panel = "_No crisis resources triggered._"

    return history, helplines_panel


def clear_all():
    return [], None, ""



CSS = """
.chatbot-title { text-align: center; font-size: 1.5rem; margin-bottom: 8px; }
.panel-box { 
    border-radius: 8px; 
    padding: 12px; 
    background: #f9f9f9; 
    color: #111111 !important; 
}
.panel-box p, .panel-box h2, .panel-box h3, .panel-box span, .panel-box em, .panel-box strong { 
    color: #111111 !important; 
}
.crisis-warning { background: #fff3f3; border-left: 4px solid #e53e3e; padding: 10px; }
footer { display: none !important; }
"""

DISCLAIMER = """
>  **Important:** This chatbot is an AI prototype and is NOT a substitute for 
> professional mental health care. If you are in crisis, please contact a licensed 
> counselor or call 9152987821 (INDIA Suicide & Crisis Lifeline) immediately.
> 
> **Transparency:** Every response includes an XAI (Explainable AI) explanation 
> showing which words influenced the emotion detection.
"""

with gr.Blocks(css=CSS, title="MITHRA : Mental Health Support Chatbot") as demo:

    gr.Markdown("# MITHRA : Mental Health Support Chatbot", elem_classes="chatbot-title")
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(
                label="Conversation",
                height=500,
                bubble_full_width=False
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Share how you're feeling...",
                    label="",
                    lines=2,
                    scale=5
                )
                with gr.Column(scale=1):
                    send_btn  = gr.Button("Send ", variant="primary")
                    clear_btn = gr.Button("Clear ")

        
        with gr.Column(scale=2):
            gr.Markdown("### Support Resources")
            helplines_panel = gr.Markdown(
                value="_Helplines will appear if needed._",
                elem_classes="panel-box"
            )

    
    send_btn.click(
        fn=respond,
        inputs=[msg_box, chatbot_ui],
        outputs=[chatbot_ui, helplines_panel]
    ).then(lambda: "", outputs=msg_box)

    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot_ui],
        outputs=[chatbot_ui, helplines_panel]
    ).then(lambda: "", outputs=msg_box)

    clear_btn.click(
        fn=clear_all,
        outputs=[chatbot_ui, msg_box, helplines_panel]
    )

    gr.Markdown("""
    ---
    **Example prompts to try:**
    - *"I've been feeling really hopeless lately and don't see a way out."*
    - *"I'm so anxious about my presentation tomorrow I can't sleep."*
    - *"My grandmother passed away last week and I don't know how to cope."*
    - *"I feel angry all the time and I don't know why."*
    """)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
