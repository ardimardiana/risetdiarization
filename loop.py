import os
import re
import subprocess
import collections
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

BUDGET = 3
BEST_DER = 0.3822 # Your baseline DER
TARGET_FILE = "agent_config.py"
LOG_FILE = "results.tsv"
MODEL_ID = "gemini-3.5-flash"

# =========================================================================
# ROOT-LEVEL TEMPLATE
# =========================================================================
AGENT_TEMPLATE = """import torch
from torch_audiomentations import Compose, PitchShift, Gain, PolarityInversion

# =========================================================================
# --- AGENT CODE INJECTED HERE ---
{AGENT_CODE}
# =========================================================================
"""

system_prompt = """You are an autonomous Machine Learning research agent. 
Minimize the Diarization Error Rate (DER) by tuning hyperparameters and audio augmentations for a Pyannote segmentation model.

Rules:
1. Output ONLY the complete Python function named `get_agent_configuration()`.
2. CRITICAL: The very first line inside your function MUST be a comment starting exactly with `# STRATEGY: ` explaining your approach.
3. You must return a tuple: `(hyperparameters_dict, augmentation_pipeline)`.
4. MAX BATCH SIZE: 16. Do not exceed a batch size of 16 to prevent CUDA OOM on the 22GB VRAM GPU.
5. You may ONLY use the following classes from `torch_audiomentations` with these EXACT argument signatures. DO NOT invent kwargs:
   - `Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=0.5)`
   - `PitchShift(min_transpose_semitones=-2.0, max_transpose_semitones=2.0, sample_rate=16000, p=0.5)`
   - `PolarityInversion(p=0.5)`
6. Output your code in a single ```python code block.

CRITICAL PARADIGM: ACOUSTIC FINE-TUNING
- Tune `learning_rate` (typically between 1e-5 and 1e-3).
- Adjust `vad_onset`, `vad_offset`, and `min_speech_duration` to control strictness.
- Use the augmentation pipeline to perturb the existing audio to prevent overfitting."""

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("iteration\tstatus\tder\tmissed\tfalse_alarm\tconfusion\tnotes\n")

failed_strategies = collections.deque(maxlen=3)

def extract_code(llm_response):
    match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_strategy(code_string):
    for line in code_string.split('\n'):
        if line.strip().startswith("# STRATEGY:"):
            return line.strip()
    return "# STRATEGY: Unknown approach"

def run_loop():
    # 1. Run Training
    print("-> Training Model...")
    train_result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    if train_result.returncode != 0:
        return {"status": "crash", "output": train_result.stderr.strip()}

    # 2. Run Evaluation
    print("-> Evaluating Model...")
    eval_result = subprocess.run(["python", "eval.py"], capture_output=True, text=True)
    if eval_result.returncode != 0 or "STATUS:CRASH" in eval_result.stdout:
        return {"status": "crash", "output": eval_result.stderr.strip() + " | " + eval_result.stdout.strip()}
    
    try:
        parts = {p.split(":")[0]: p.split(":")[1] for p in eval_result.stdout.strip().split("|")}
        return {
            "status": "success",
            "der": float(parts["DER"]),
            "ms": float(parts["MS"]),
            "fa": float(parts["FA"]),
            "cf": float(parts["CF"])
        }
    except Exception as e:
        return {"status": "crash", "output": f"Parse failed. Raw: {eval_result.stdout.strip()}"}

last_agent_code = """def get_agent_configuration():
    # STRATEGY: Conservative baseline to verify the pipeline runs.
    hyperparameters = {
        "learning_rate": 1e-4,
        "batch_size": 8,
        "vad_onset": 0.5,     
        "vad_offset": 0.5,
        "min_speech_duration": 0.1
    }
    augmentation_pipeline = Compose(
        transforms=[Gain(min_gain_in_db=-3.0, max_gain_in_db=3.0, p=0.5)]
    )
    return hyperparameters, augmentation_pipeline
"""
last_feedback = f"First run initialized. Baseline DER target is {BEST_DER}."

# MAIN LOOP
for i in range(1, BUDGET + 1):
    print(f"\n--- Iteration {i}/{BUDGET} ---")
    print(f"Agent ({MODEL_ID}) is thinking...")
    
    history_text = "\n".join(failed_strategies) if failed_strategies else "None yet."
    user_prompt = f"Feedback:\n{last_feedback}\n\nFailed approaches:\n{history_text}\n\nCurrent best code:\n```python\n{last_agent_code}\n```"
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID, contents=user_prompt, config=types.GenerateContentConfig(system_instruction=system_prompt)
        )
        new_agent_code = extract_code(response.text)
    except Exception as e:
        print(f"API Error: {e}")
        continue
        
    if not new_agent_code:
        print("LLM failed to output valid code. Skipping.")
        continue
        
    full_new_code = AGENT_TEMPLATE.replace("{AGENT_CODE}", new_agent_code)
    with open(TARGET_FILE, "w") as f:
        f.write(full_new_code)
        
    eval_metrics = run_loop()
    
    if eval_metrics["status"] == "success":
        current_der = eval_metrics["der"]
        strategy_used = extract_strategy(new_agent_code)
        print(f"Strategy: {strategy_used}\nResult -> DER: {current_der:.4f}")
        
        if current_der < BEST_DER:
            print("🟢 IMPROVEMENT! Keeping changes.")
            BEST_DER = current_der
            last_agent_code = new_agent_code 
            subprocess.run(["git", "commit", "-am", f"Iter {i}: DER {current_der:.4f}"])
            last_feedback = f"Improved DER to {current_der:.4f}. Missed: {eval_metrics['ms']:.4f}, FA: {eval_metrics['fa']:.4f}, CF: {eval_metrics['cf']:.4f}."
            log_line = f"{i}\tkeep\t{current_der}\t{eval_metrics['ms']}\t{eval_metrics['fa']}\t{eval_metrics['cf']}\tImproved\n"
        else:
            print("🔴 REGRESSION. Reverting changes.")
            subprocess.run(["git", "checkout", TARGET_FILE])
            failed_strategies.append(f"- {strategy_used} -> FAILED (DER: {current_der:.4f})")
            last_feedback = f"Worsened DER to {current_der:.4f}. Adjust learning rate or VAD offsets."
            log_line = f"{i}\trevert\t{current_der}\t{eval_metrics['ms']}\t{eval_metrics['fa']}\t{eval_metrics['cf']}\tRegression\n"
            
    else:
        print("💥 CRASH. Reverting changes.")
        clean_output = eval_metrics['output'].replace('\n', ' ')
        print(f"Crash Reason: {clean_output}") 
        subprocess.run(["git", "checkout", TARGET_FILE])
        strategy_used = extract_strategy(new_agent_code)
        failed_strategies.append(f"- {strategy_used} -> CRASHED: {clean_output[:100]}")
        last_feedback = f"Your code crashed: {clean_output[:200]}. Ensure proper dict formatting."
        log_line = f"{i}\tcrash\t-\t-\t-\t-\t{clean_output}\n"

    with open(LOG_FILE, "a") as f:
        f.write(log_line)

print("\nBudget exhausted. Experiment complete.")