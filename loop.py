import os
import re
import subprocess
import collections
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

BUDGET = 10
BEST_DER = 0.3822
TARGET_FILE = "extractor.py"
EVAL_CMD = ["python", "eval.py"]
LOG_FILE = "results.tsv"
#MODEL_ID = "gemini-3.1-flash-lite-preview"
#MODEL_ID = "gemini-2.5-pro"
MODEL_ID = "gemini-3.5-flash"

# =========================================================================
# 1. ROOT-LEVEL TEMPLATE
# =========================================================================
EXTRACTOR_TEMPLATE = """import librosa
import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline
import os

# =========================================================================
# --- AGENT CODE INJECTED HERE ---
{AGENT_CODE}
# =========================================================================

def process_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Isolate DSP logic in the agent's function
    y_masked = apply_dsp_filter(y, sr)
    
    temp_wav = "temp_masked_audio.wav"
    sf.write(temp_wav, y_masked, sr)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=os.environ.get("HF_TOKEN")
    )
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    out = pipeline(temp_wav)
    
    diarization = out.speaker_diarization if hasattr(out, "speaker_diarization") else out
    
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
        
    return diarization
"""

# =========================================================================
# 2. SYSTEM PROMPT & KEY INFORMATION (DSP HINTS)
# =========================================================================
system_prompt = """You are an autonomous DSP research agent. 
Minimize the Diarization Error Rate (DER) of an audio file containing speech.

Rules:
1. Output ONLY a complete Python function named `apply_dsp_filter(y, sr)`.
2. CRITICAL: The very first line inside your function MUST be a comment starting exactly with `# STRATEGY: ` explaining your approach in 1 short sentence.
3. `y` is a 1D numpy array (mono audio), `sr` is the sample rate. Return the modified 1D numpy array `y_masked`.
4. Ensure all variables are defined inside your function. Prevent dimensional crashes (e.g., use `.flatten()`).
5. DO NOT use ANY Pyannote functions. You are ONLY processing raw numpy arrays.
6. Output your code in a single ```python code block.

HINTS FOR SUCCESS (AVOID DIARIZATION ARTIFACTS):
- Avoid Hard Masking: Setting arrays abruptly to 0.0 creates artifacts (clicks) that destroy embeddings, spiking False Alarms (FA) and Confusion (CF). Use Soft Masking (e.g., attenuating by multiplying by 0.2) or Wiener filters.
- Try HPSS: Laughter often has strong percussive/broadband energy, while speech is highly harmonic. Use `librosa.effects.hpss` to separate or suppress percussive components naturally.
- Use Smooth Transitions: If using thresholds, apply a Hann window or a long convolution filter to smooth the transitions between masked and unmasked regions."""

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("iteration\tstatus\tder\tmissed\tfalse_alarm\tconfusion\tnotes\n")

# =========================================================================
# 3. HELPER FUNCTIONS & MEMORY QUEUE
# =========================================================================
failed_strategies = collections.deque(maxlen=3) # Limit memory to last 3 failures

def extract_code(llm_response):
    match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_strategy(code_string):
    for line in code_string.split('\n'):
        if line.strip().startswith("# STRATEGY:"):
            return line.strip()
    return "# STRATEGY: Unknown approach"

def run_eval():
    result = subprocess.run(EVAL_CMD, capture_output=True, text=True)
    if result.returncode != 0 or "STATUS:CRASH" in result.stdout:
        return {"status": "crash", "output": result.stderr.strip() + " | " + result.stdout.strip()}
    
    try:
        parts = {p.split(":")[0]: p.split(":")[1] for p in result.stdout.strip().split("|")}
        return {
            "status": "success",
            "der": float(parts["DER"]),
            "ms": float(parts["MS"]),
            "fa": float(parts["FA"]),
            "cf": float(parts["CF"])
        }
    except Exception as e:
        return {"status": "crash", "output": f"Failed to parse eval output. Raw: {result.stdout.strip()}"}

# =========================================================================
# 4. INITIAL SEED (Must include # STRATEGY:)
# =========================================================================
last_agent_code = """def apply_dsp_filter(y, sr):
    # STRATEGY: Baseline thresholding using MFCC variance and ZCR.
    hop_length = 512
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    mfcc_var = np.var(mfccs, axis=0)
    
    instability_threshold = np.mean(mfcc_var) + (1.5 * np.std(mfcc_var))
    zcr_threshold = np.mean(zcr) + (2.0 * np.std(zcr))
    
    is_laughter = (mfcc_var > instability_threshold) & (zcr > zcr_threshold)
    mask = np.where(is_laughter, 0.5, 1.0) # Soft mask baseline
    smooth_mask = np.convolve(mask, np.ones(5)/5, mode='same')
    
    expanded_mask = np.repeat(smooth_mask, hop_length)
    expanded_mask = expanded_mask[:len(y)]
    
    y_masked = (y * expanded_mask).flatten()
    return y_masked
"""

last_feedback = f"This is the first run. The current baseline DER is {BEST_DER}."

# =========================================================================
# 5. MAIN LOOP
# =========================================================================
for i in range(1, BUDGET + 1):
    print(f"\n--- Iteration {i}/{BUDGET} ---")
    print(f"Agent ({MODEL_ID}) is thinking...")
    
    history_text = "\n".join(failed_strategies) if failed_strategies else "None yet."
    
    user_prompt = f"""Feedback from last run:
{last_feedback}

Failed approaches to AVOID (History):
{history_text}

Current best DSP logic:
```python
{last_agent_code}
```"""
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        new_agent_code = extract_code(response.text)
    except Exception as e:
        print(f"API Error: {e}")
        continue
        
    if not new_agent_code:
        print("LLM failed to output valid code. Skipping iteration.")
        continue

    # LINTER: Block forbidden code
    if "get_timeline" in new_agent_code or "pyannote" in new_agent_code.lower():
        print("🛑 PRE-FLIGHT REJECTION: AI generated forbidden Pyannote attributes.")
        last_feedback = "CRITICAL ERROR: Code contained Pyannote references. Return a 1D numpy array ONLY."
        with open(LOG_FILE, "a") as f:
            f.write(f"{i}\treject\t-\t-\t-\t-\tGenerated forbidden attribute\n")
        continue
        
    # INJECT & WRITE
    full_new_code = EXTRACTOR_TEMPLATE.replace("{AGENT_CODE}", new_agent_code)
    with open(TARGET_FILE, "w") as f:
        f.write(full_new_code)
        
    print("Running evaluation...")
    eval_metrics = run_eval()
    
    if eval_metrics["status"] == "success":
        current_der = eval_metrics["der"]
        strategy_used = extract_strategy(new_agent_code)
        print(f"Strategy: {strategy_used}")
        print(f"Result -> DER: {current_der:.4f}")
        
        if current_der < BEST_DER:
            print("🟢 IMPROVEMENT! Keeping changes.")
            BEST_DER = current_der
            last_agent_code = new_agent_code 
            subprocess.run(["git", "commit", "-am", f"Iter {i}: DER {current_der:.4f}"])
            last_feedback = f"Great! Improved DER to {current_der:.4f}. Missed: {eval_metrics['ms']:.4f}, FA: {eval_metrics['fa']:.4f}, CF: {eval_metrics['cf']:.4f}."
            log_line = f"{i}\tkeep\t{current_der}\t{eval_metrics['ms']}\t{eval_metrics['fa']}\t{eval_metrics['cf']}\tImproved\n"
        else:
            print("🔴 REGRESSION. Reverting changes & Saving to Memory.")
            subprocess.run(["git", "checkout", TARGET_FILE])
            
            # Save to fail memory
            fail_record = f"- {strategy_used} -> FAILED (DER: {current_der:.4f}, FA: {eval_metrics['fa']:.4f}, CF: {eval_metrics['cf']:.4f})"
            failed_strategies.append(fail_record)
            
            last_feedback = f"Worsened DER to {current_der:.4f}. Avoid this approach. Check if FA or CF spiked due to harsh masking artifacts."
            log_line = f"{i}\trevert\t{current_der}\t{eval_metrics['ms']}\t{eval_metrics['fa']}\t{eval_metrics['cf']}\tRegression\n"
            
    else:
        print("💥 CRASH. Reverting changes.")
        clean_output = eval_metrics['output'].replace('\n', ' ')
        print(f"Crash Reason: {clean_output}") 
        
        subprocess.run(["git", "checkout", TARGET_FILE])
        
        # Penanganan crash pada memory
        strategy_used = extract_strategy(new_agent_code)
        failed_strategies.append(f"- {strategy_used} -> CRASHED: {clean_output[:100]}")
        
        last_feedback = f"Your code caused a crash: {clean_output[:200]}. Ensure `apply_dsp_filter` returns a valid 1D array."
        log_line = f"{i}\tcrash\t-\t-\t-\t-\t{clean_output}\n"

    with open(LOG_FILE, "a") as f:
        f.write(log_line)

print("\nBudget exhausted. Experiment complete.")