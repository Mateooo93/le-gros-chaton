# Le Gros Chaton

 **A 9B coding agent built on Qwen3.5-9B.** 25% on Terminal-Bench 2.0.
 Model: <https://huggingface.co/mateo0093/le-gros-chaton>

## DEMO

demo: <https://mateooo93.github.io/le-gros-chaton/>

Le gros chaton is a model trained on top of Qwen3.5-9B using fable 5 traces datasets aswell as kimi k3 traces. the goal was to acheive a good coding assistant model so it was also trained on the pi harness for tool calling, when it made a mistake we corrected that and fed it to the training pipeline, the final result wasn't exactly what i was looking for but I think this was a good learning opportunity. I will attempt this once again with a completely different approach and taking into account the mistakes i could have avoided and the things i learned.
here is the adapters aswell as the merged model.

| Step | Adapter | Purpose |
|---|---|---|
| 1 | [Fable5](https://huggingface.co/mateo0093/le-gros-chaton-qwen) | Tool-call format alignment (91.2% adapter) |
| 2 | [Trajectory SFT (16K)](https://huggingface.co/mateo0093/le-gros-chaton-qwen-traj-sft-16k) |agentic trace imitation on TB-2.0 (474 verified traces, 180 steps) |
| 3 | *(merged)* | `base + Fable5 + traj → merged-16k` (12 hybrid target modules) |


## Self-host
if you want to self host with vllm on your own gpu:
```bash
vllm serve mateo0093/le-gros-chaton \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768
```

Or with `transformers` directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mateo0093/le-gros-chaton",
    torch_dtype="bfloat16",
    device_map="cuda:0",
)
tok = AutoTokenizer.from_pretrained("mateo0093/le-gros-chaton")
```

## Benchmark — Terminal-Bench 2.0

5 tasks × 5 attempts each (leaderboard protocol):

| Task | Pass rate |
|---|---|
| fix-git | 3/5 |
| log-summary-date-ranges | 2/5 |
| overfull-hbox | 1/5 |
| regex-log | 0/5 |
| count-dataset-tokens | 0/5 |
| **Total** | **6/25 = 25%** |

25% is a good score for a 9b model but the model still ma
