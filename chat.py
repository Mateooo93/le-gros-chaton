"""Interactive chat using the unified InferenceEngine.

Simplified to a thin wrapper — all generation logic lives in ``inference.py``.
"""
import sys
from inference import InferenceEngine


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
    engine = InferenceEngine(ckpt)

    print("Chat with Chaton.  Type 'quit' to exit, /clear to reset cache.\n")
    engine.clear_cache()
    history: list[dict] = [
        {"role": "system", "content": "You are a helpful coding assistant."}
    ]

    while True:
        user = input(">>> ")
        if user.strip().lower() in ("quit", "exit"):
            break
        if user.strip() == "/clear":
            engine.clear_cache()
            history = history[:1]
            print("(cache reset)\n")
            continue
        if not user.strip():
            continue

        history.append({"role": "user", "content": user})
        reply = engine.chat(history)
        print(f"  {reply}\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()