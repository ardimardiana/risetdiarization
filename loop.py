import os
import re
import subprocess
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

BUDGET = 5
BEST_DER = 0.3822
TARGET_FILE = "extractor.py"
EVAL_CMD = ["python", "eval.py"]
LOG_FILE = "results.tsv"
MODEL_ID = "gemini-3.1-flash-lite-preview"
#MODEL_ID = "gemini-2.5-pro"
#MODEL_ID = "gemini-3.5-flash"

# 1. ROOT-LEVEL TEMPLATE
# The LLM's function replaces {AGENT_CODE} at the root indentation level.
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
    
    # Run Diarization
    out = pipeline(temp_wav)
    
    # Unwrap the Annotation object to prevent 'get_timeline' crashes
    # This gracefully handles both old and new pyannote versions
    diarization = out.speaker_diarization if hasattr(out, "speaker_diarization") else out
    
    # Cleanup
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
        
    return diarization
"""

# 2. STRICT FUNCTION-BASED PROMPT
system_prompt = """You are an autonomous DSP research agent. 
Minimize the Diarization Error Rate (DER) of an audio file containing speech and heavy laughter.

Rules:
1. Output ONLY a complete Python function named `apply_dsp_filter(y, sr)`. Do not output imports.
2. `y` is a 1D numpy array (mono audio), `sr` is the sample rate.
3. Return the modified 1D numpy array `y_masked`.
4. Ensure all variables are defined inside your function.
5. Prevent dimensional crashes: ensure your final mask and `y_masked` are strictly 1D arrays (e.g., use `.flatten()`).
6. CRITICAL: DO NOT use `get_timeline()`, `DiarizeOutput`, or ANY Pyannote functions. You are ONLY processing raw numpy arrays.
7. LIBROSA GUARDRAILS: Double-check your kwargs. `librosa.feature.spectral_centroid` uses `n_fft` and `hop_length`, NOT `frame_length`. `librosa.feature.rms` and `zero_crossing_rate` use `frame_length`.
8. Output your code in a single ```python code block."""

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("iteration\tstatus\tder\tmissed\tfalse_alarm\tconfusion\tnotes\n")

def extract_code(llm_response):
    match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
    # Strip whitespace to prevent root-level IndentationErrors
    return match.group(1).strip() if match else None

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

# 3. SEED WITH A COMPLETE FUNCTION
last_agent_code = """def apply_dsp_filter(y, sr):
    hop_length = 512
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    mfcc_var = np.var(mfccs, axis=0)
    
    instability_threshold = np.mean(mfcc_var) + (1.5 * np.std(mfcc_var))
    zcr_threshold = np.mean(zcr) + (2.0 * np.std(zcr))
    
    is_laughter = (mfcc_var > instability_threshold) & (zcr > zcr_threshold)
    mask = np.where(is_laughter, 0.0, 1.0)
    smooth_mask = np.convolve(mask, np.ones(5)/5, mode='same')
    
    expanded_mask = np.repeat(smooth_mask, hop_length)
    expanded_mask = expanded_mask[:len(y)]
    
    # Force 1D shape to prevent soundfile crash
    y_masked = (y * expanded_mask).flatten()
    return y_masked
"""

last_feedback = "This is the first run. The current baseline DER is 0.3822."

for i in range(1, BUDGET + 1):
    print(f"\n--- Iteration {i}/{BUDGET} ---")
    print(f"Agent ({MODEL_ID}) is thinking...")
    
    user_prompt = f"Feedback from last run:\n{last_feedback}\n\nCurrent DSP logic:\n```python\n{last_agent_code}\n```"
    
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

    # ==========================================
    # PRE-EXECUTION LINTER: Block forbidden code
    # ==========================================
    if "get_timeline" in new_agent_code or "pyannote" in new_agent_code.lower():
        print("🛑 PRE-FLIGHT REJECTION: AI generated forbidden Pyannote attributes. Skipping eval.")
        last_feedback = "CRITICAL ERROR: Your code contained `get_timeline` or Pyannote references. Do NOT interact with Pyannote objects. Return a 1D numpy array."
        log_line = f"{i}\treject\t-\t-\t-\t-\tGenerated forbidden attribute\n"
        with open(LOG_FILE, "a") as f:
            f.write(log_line)
        continue
    # ==========================================
        
    # DIRECT REPLACEMENT
    full_new_code = EXTRACTOR_TEMPLATE.replace("{AGENT_CODE}", new_agent_code)
    
    with open(TARGET_FILE, "w") as f:
        f.write(full_new_code)
        
    print("Running evaluation...")
    eval_metrics = run_eval()
    
    if eval_metrics["status"] == "success":
        current_der = eval_metrics["der"]
        print(f"Result -> DER: {current_der:.4f}")
        
        if current_der < BEST_DER:
            print("🟢 IMPROVEMENT! Keeping changes.")
            BEST_DER = current_der
            last_agent_code = new_agent_code 
            subprocess.run(["git", "commit", "-am", f"Iter {i}: Improved DER to {current_der:.4f}"])
            last_feedback = f"Great! Improved DER to {current_der:.4f}. Missed Speech: {eval_metrics['ms']:.4f}, False Alarm: {eval_metrics['fa']:.4f}."
            log_line = f"{i}\tkeep\t{current_der}\t{eval_metrics['ms']}\t{eval_metrics['fa']}\t{eval_metrics['cf']}\tImproved\n"
        else:
            print("🔴 REGRESSION. Reverting changes.")
            subprocess.run(["git", "checkout", TARGET_FILE])
            last_feedback = f"Worsened DER to {current_der:.4f} (Baseline to beat is {BEST_DER:.4f}). Try a different strategy."
            log_line = f"{i}\trevert\t{current_der}\t{eval_metrics['ms']}\t{eval_metrics['fa']}\t{eval_metrics['cf']}\tRegression\n"
            
    else:
        print("💥 CRASH. Reverting changes.")
        clean_output = eval_metrics['output'].replace('\n', ' ')
        print(f"Crash Reason: {clean_output}") 
        
        subprocess.run(["git", "checkout", TARGET_FILE])
        last_feedback = f"Your code caused a crash: {clean_output}. Ensure `apply_dsp_filter` returns a 1D array."
        log_line = f"{i}\tcrash\t-\t-\t-\t-\t{clean_output}\n"

    with open(LOG_FILE, "a") as f:
        f.write(log_line)

print("\nBudget exhausted. Experiment complete.")