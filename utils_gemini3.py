import os
import base64
import io
from pdf2image import convert_from_path
from PIL import Image
import openai
import json

def pdf_pages_to_resized_base64(pdf_path, target_width=1280, target_height=720, fmt="JPEG"):
    """
    Convert each page of a PDF into an image at a target resolution, then encode as Base64.

    Args:
        pdf_path: PDF file path
        target_width: Target width (pixels). If provided, scale by width
        target_height: Target height (pixels). If provided, scale by height (width takes priority)
        fmt: Image format (PNG/JPG)

    Returns:
        List where each element is the Base64-encoded string of a single page image
    """
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
        
        # Convert at default DPI (200) first to get initial image sizes.
        # Note: this DPI is only for the initial render; final size is controlled by target_width/height.
        images = convert_from_path(pdf_path, dpi=200, fmt=fmt.lower())
        # print(f"PDF has {len(images)} pages. Resizing to target resolution...")
        
        base64_list = []
        for img in images[:10]:
            # Get original size
            orig_width, orig_height = img.size
            
            # Compute scaling factor (prefer width; otherwise use height)
            if target_width:
                scale = target_width / orig_width
            elif target_height:
                scale = target_height / orig_height
            else:
                # If neither is provided, keep original size
                scale = 1.0
            
            # Compute new size (ensure integer pixels)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            # print(f"Resizing: {orig_width}×{orig_height} → {new_width}×{new_height}")
            
            # Resize using high-quality resampling
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Encode as Base64
            buffer = io.BytesIO()
            resized_img.save(buffer, format=fmt)
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            base64_list.append(img_base64)
        
        return base64_list
    
    except Exception as e:
        print(f"PDF processing failed: {str(e)}")
        return None
    
import requests
import time
max_retries = 3
retry_delay = 2  # seconds

url = 'https://gpt-us.singularity-ai.com/gpt-proxy/chat/completions'
# start_hour, end_hour are hour-level time parameters, format: 2025111916
# curl -H "Authorization: Bearer gpt-bf3ea3b06e5c1cf68b5a8b609dbe" 'https://gpt-hk.singularity-ai.com/gpt-proxy/key/detail'
app_key = os.environ.get('GPTAppKey', 'gpt-bf3ea3b06e5c1cf68b5a8b609dbe')        # Replace this with your APPKey
if app_key == '':
    print("not set env var GPTAppKey")
    exit(1)
from openai import OpenAI
headers = {
    "app_key": app_key,
    "Content-Type": "application/json"
}

if app_key == '':
    print("not set env var GPTAppKey")
    exit(1)
client = OpenAI(
    api_key=app_key,
    base_url="https://gpt-us.singularity-ai.com/gpt-proxy/google"
)

def call_gpt_api(
    system_prompt,  # NEW: pass in the Agent role/system prompt
    question, 
    base64_images=[], 
    model="gemini-3-pro-preview", # Recommended: use gpt-4o for images
    temperature=0.01, # Lower randomness
    max_tokens=4096
):
    """
    Corrected API call function
    """
    # Configure API (your custom configuration)
    app_key = os.environ.get('GPTAppKey', 'gpt-bf3ea3b06e5c1cf68b5a8b609dbe')        # Replace this with your APPKey
    # 1. Build the system message
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # 2. Build the user message (mixed images + text)
    user_content = []
    
    # Add all images first
    for img_base64 in base64_images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        })
    
    # Add the question text once at the end (Fix: do NOT put this inside the loop)
    user_content.append({
        "type": "text",
        "text": question
    })
    
    messages.append({"role": "user", "content": user_content})
    config_args = {
            # "temperature": temperature,
            "max_output_tokens": max_tokens
        }
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                n=1,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content.strip()
            return answer
            
        except requests.exceptions.Timeout:
            print(f"[!] Request timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                print(f"Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
                
        except requests.exceptions.HTTPError as e:
            print(f"[!] HTTP error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
                
        except requests.exceptions.RequestException as e:
            print(f"[!] Network request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
                
        except KeyError as e:
            print(f"[!] Response format error (attempt {attempt + 1}/{max_retries}): missing key {e}")
            if attempt < max_retries - 1:
                print(f"Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)
                
        except Exception as e:
            print(f"[!] Unknown error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay)

    # Return empty dict after all retries fail
    print("[!] All retry attempts failed; returning empty dict")
    return {}
        
    
    
import json
import re
from typing import Dict, Any, Optional


def parse_decision(raw_output: str) -> Dict[str, Any]:
    """
    Parse the model's decision output. Supports two formats:
    1) A complete JSON string
    2) Non-JSON natural language text (extract key fields via regex)

    Returns a unified dict format. Missing fields are filled with None.
    """
    raw_output = raw_output.strip()
    
    # 1. Try parsing JSON directly (including code-fenced outputs)
    if raw_output.startswith("```json") or raw_output.startswith("```"):
        # Remove code fences
        raw_output = re.sub(r"^```(?:json)?\s*|```$", "", raw_output, flags=re.MULTILINE).strip()
    
    # Try parsing JSON again (may contain extra newlines/spaces)
    try:
        data = json.loads(raw_output)
        # Basic validation
        if not isinstance(data, dict):
            raise ValueError("JSON root is not an object")
        return _normalize_result(data)
    except json.JSONDecodeError:
        # If JSON parsing fails, fall back to regex
        pass
    
    # 2. Regex fallback (very effective for non-JSON outputs)
    result = {
        "final_decision": None,
        "final_score": None,
        "decision_archetype": None,
        "score_interpretation": None,
        "key_factor": None,
        "confidence": None
    }
    
    # final_decision
    decision_match = re.search(
        r'"final_decision"\s*:\s*"([^"]+)"|'               # JSON-style
        r'final[_ ]?decision[:：]\s*(Oral|Spotlight|Poster|Reject)', 
        raw_output, re.IGNORECASE
    )
    if decision_match:
        result["final_decision"] = (decision_match.group(1) or decision_match.group(2)).strip()
        
    # final_score
    score_match = re.search(
        r'"final_score"\s*:\s*"([^"]+)"|'               # JSON-style
        r'final[_ ]?score[:：]\s*(Oral|Spotlight|Poster|Reject)', 
        raw_output, re.IGNORECASE
    )
    if score_match:
        result["final_score"] = (score_match.group(1) or score_match.group(2)).strip()
    
    # decision_archetype (many enum values; joined by |)
    archetype_pattern = (
        "Polarized_Saved_by_Champion|Vanilla_Rejection|Uncontested_Success|"
        "Flawed_but_Novel|Fatal_Technical_Reject"
    )
    arch_match = re.search(
        rf'"decision_archetype"\s*:\s*"([^"]+)"|' 
        rf'decision[_ ]?archetype[:：]\s*({archetype_pattern})',
        raw_output, re.IGNORECASE
    )
    if arch_match:
        result["decision_archetype"] = (arch_match.group(1) or arch_match.group(2)).strip()
    
    # score_interpretation (usually a longer explanation)
    interp_match = re.search(
        r'"score_interpretation"\s*:\s*"([^"]*(?:\\"[^"]*)*)"|'  # handle escaped quotes
        r'(?:score[_ ]?interpretation|score explanation)[:：]\s*([^\n"}]+)', 
        raw_output, re.IGNORECASE
    )
    if interp_match:
        text = interp_match.group(1) or interp_match.group(2) or ""
        # Simple cleanup for trailing quotes/spaces
        result["score_interpretation"] = text.strip().strip('"').strip()
    
    # key_factor (short phrase)
    key_match = re.search(
        r'"key_factor"\s*:\s*"([^"]+)"|' 
        r'(?:key[_ ]?factor|decisive factor)[:：]\s*([^\n"}]+)',
        raw_output, re.IGNORECASE
    )
    if key_match:
        result["key_factor"] = (key_match.group(1) or key_match.group(2)).strip().strip('"')
    
    # confidence
    conf_match = re.search(
        r'"confidence"\s*:\s*"([^"]+)"|' 
        r'confidence[:：]\s*(High|Medium|Low)',
        raw_output, re.IGNORECASE
    )
    if conf_match:
        result["confidence"] = (conf_match.group(1) or conf_match.group(2)).strip()
    
    return _normalize_result(result)


def _normalize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Unify field names and apply basic validation/cleanup."""
    normalized = {
        "final_decision": _clean_str(data.get("final_decision")),
        "final_score": _clean_str(data.get("final_score")),
        "decision_archetype": _clean_str(data.get("decision_archetype")),
        "score_interpretation": _clean_str(data.get("score_interpretation")),
        "key_factor": _clean_str(data.get("key_factor")),
        "confidence": _clean_str(data.get("confidence")),
    }
    
    # Optional: strictly validate value ranges
    valid_decisions = {"Oral", "Spotlight", "Poster", "Reject"}
    if normalized["final_decision"] not in valid_decisions and normalized["final_decision"]:
        normalized["final_decision"] = None
    
    valid_archetypes = {
        "Polarized_Saved_by_Champion", "Vanilla_Rejection",
        "Uncontested_Success", "Flawed_but_Novel", "Fatal_Technical_Reject"
    }
    if normalized["decision_archetype"] not in valid_archetypes and normalized["decision_archetype"]:
        normalized["decision_archetype"] = None
        
    valid_conf = {"High", "Medium", "Low"}
    if normalized["confidence"] not in valid_conf and normalized["confidence"]:
        normalized["confidence"] = None
        
    return normalized


def _clean_str(s: Any) -> Optional[str]:
    if s is None:
        return None
    if isinstance(s, str):
        return s.strip() or None
    return str(s).strip() or None


# ====================== Usage Example ======================

if __name__ == "__main__":
    # Case 1: Valid JSON
    s1 = '''```json
    {
      "final_decision": "Spotlight",
      "decision_archetype": "Polarized_Saved_by_Champion",
      "score_interpretation": "Scores suggest mediocrity but strong champion saves it.",
      "key_factor": "Champion enthusiasm",
      "confidence": "High"
    }
    ```'''
    
    # Case 2: Messy plain-text output
    s2 = """
    我的决定是 Spotlight。
    decision_archetype: Flawed_but_Novel
    分数解释：平均分不高，但创新性很强，有一个 reviewer 给了8分。
    key_factor: Novelty outweighed technical flaws
    confidence: Medium
    """
    
    print(parse_decision(s1))
    print(parse_decision(s2))
