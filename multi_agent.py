import json
import os
from typing import List, Dict, Any
from openai import OpenAI
from prompt import *
# from prompt_mind import *
# from utils import *
from utils_gemini3 import *

# =============================================================================
# CONFIGURATION
# =============================================================================
# 3. Run tasks with multi-threading
# max_workers can be adjusted based on CPU cores or network I/O.
# For I/O-bound workloads, you can set it higher (e.g., 8 or 16).
max_workers = 1
success_count = 0

# Replace with your API key
client = OpenAI(api_key="sk-......")

# MODEL_NAME = "gemini-2.5-pro"  # Recommended: use gpt-4o for best reasoning + JSON reliability
MODEL_NAME = "gemini-3-pro-preview"  # Recommended: use gpt-4o for best reasoning + JSON reliability

BASE_DIR = "iclr2025_bench/iclr2025_bench"
RESULT_DIR = os.path.join(BASE_DIR, "results_prompt_gemini3_theory_of_mind")
os.makedirs(RESULT_DIR, exist_ok=True)

categories = ['3D', 'MLLM', 'RL']

# =============================================================================
# DATA PREPROCESSING (KEY STEP: CLEAN YOUR JSON)
# =============================================================================

def extract_discussion_thread(comments_tree: List[Dict]) -> str:
    """
    Recursively extract dialogue from the comment tree, tagging who said what,
    forming a clear chronological timeline.
    """
    discussion_text = ""

    for comment in comments_tree:
        role = comment.get('type', 'unknown')  # author / reviewer
        content = comment.get('content', {}).get('comment', '')

        # Truncate overly long replies to save tokens
        discussion_text += f"\n--- [{role.upper()} Reply] ---\n{content[:800]}...\n"

        # Recursively process child comments (reviewer follow-ups or author replies)
        if 'children' in comment and comment['children']:
            discussion_text += extract_discussion_thread(comment['children'])

    return discussion_text


def preprocess_paper_json(raw_data: Dict) -> Dict:
    """
    Convert the raw, complex JSON into structured data needed by the agents.
    """
    # 1. Basic information
    processed = {
        "title": raw_data.get("title", ""),
        "abstract": raw_data.get("abstract", "") or raw_data.get("content", {}).get("abstract", ""),
        # Note: if JSON only contains the abstract but not the full paper,
        # the content evaluator can only assess based on the abstract.
        "reviews": [],
        "global_rebuttal": ""
    }

    # 2. Extract global rebuttal (response to all reviewers)
    if "official_comment" in raw_data:
        for comment in raw_data["official_comment"]:
            if "Response to all reviewers" in comment.get("title", ""):
                processed["global_rebuttal"] = comment.get("comment", {}).get("comment", "")
                break

    # 3. Extract reviewer info and corresponding interactions
    if "peer_discussion" in raw_data:
        for idx, item in enumerate(raw_data["peer_discussion"]):
            review_content = item.get("review", {})

            # Extract reviewer ID (some data may not include explicit IDs; use index as fallback)
            reviewer_id = f"Reviewer_{idx+1}"

            # Extract interaction history (rebuttal tree)
            comments_tree = item.get("comments_tree", [])
            discussion_history = extract_discussion_thread(comments_tree)

            # Determine whether the reviewer replied (used for ghosted detection)
            has_reviewer_replied = "--- [REVIEWER Reply] ---" in discussion_history

            processed["reviews"].append({
                "id": reviewer_id,
                "score": review_content.get("rating", 0),  # ensure int-like
                "confidence": review_content.get("confidence", 0),
                "summary": review_content.get("summary", ""),
                "weaknesses": review_content.get("weaknesses", ""),
                "strengths": review_content.get("strengths", ""),
                "discussion_history": discussion_history,
                "has_reviewer_replied": has_reviewer_replied
            })

    return processed


def _normalize_decision_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate fields."""
    # Normalize final_decision
    fd = str(d.get("final_decision", "")).strip()
    final_decision_map = {
        "oral": "Oral",
        "spotlight": "Spotlight",
        "poster": "Poster",
        "reject": "Reject",
        "accepted (oral)": "Oral",
        "accepted (spotlight)": "Spotlight",
        "accepted": "Poster",  # conservative fallback
    }
    normalized_fd = final_decision_map.get(fd.lower(), fd)  # keep original if not matched

    # Normalize decision_archetype
    da = str(d.get("decision_archetype", "")).strip()
    valid_archetypes = {
        "polarized_saved_by_champion",
        "vanilla_rejection",
        "uncontested_success",
        "flawed_but_novel",
        "fatal_technical_reject"
    }
    if da.lower().replace(" ", "_") in valid_archetypes:
        da = da.lower().replace(" ", "_").title().replace("_", "_")  # keep snake_case as in spec

    # Normalize confidence
    conf = str(d.get("confidence", "")).strip().lower()
    if conf in {"high", "medium", "low"}:
        conf = conf.capitalize()
    else:
        conf = "Medium"  # default

    return {
        "final_decision": normalized_fd if normalized_fd in ["Oral", "Spotlight", "Poster", "Reject"] else "Poster",
        "decision_archetype": da if da.lower().replace("_", "") in [v.replace("_", "") for v in valid_archetypes] else "Vanilla_Rejection",
        "score_interpretation": str(d.get("score_interpretation", "")).strip() or "No interpretation provided.",
        "key_factor": str(d.get("key_factor", "")).strip() or "No key factor identified.",
        "confidence": conf
    }


# =============================================================================
# AGENT ORCHESTRATION
# =============================================================================

def run_multimodal_pipeline(json_line_data, pdf_file_data):
    # 1. Prepare data
    paper_title = json_line_data.get("title", "Unknown")

    print(f"\n[*] Processing: {paper_title}")

    # ==========================================
    # AGENT 1: VISUAL CONTENT EVALUATOR
    # ==========================================
    # Temporary file path (if needed)

    images_base64 = pdf_pages_to_resized_base64(pdf_file_data)

    # Call vision API
    agent1_res = call_gpt_api(
        system_prompt=PROMPT_CONTENT_EVALUATOR_VISION,
        question="Analyze these paper pages. Extract claims and evaluate novelty based on visual and textual evidence.",
        base64_images=images_base64,  # pass images
        model=MODEL_NAME
    )

    # ==========================================
    # AGENT 2 & 3: TEXT-BASED AGENTS (No Images)
    # ==========================================
    # Preprocess text data (reuse previous logic)
    processed_data = preprocess_paper_json(json_line_data)

    reviews_input = json.dumps([{k: v for k, v in r.items() if k != 'discussion_history'} for r in processed_data['reviews']])
    agent2_res = call_gpt_api(
        system_prompt=PROMPT_REVIEW_SYNTHESIZER,
        question=f"Analyze these reviews: {reviews_input}",
        base64_images=[],  # no images
        model=MODEL_NAME
    )

    rebuttal_input = json.dumps({
        "global_rebuttal": processed_data['global_rebuttal'],
        "review_threads": [{"id": r['id'], "score": r['score'], "history": r['discussion_history']} for r in processed_data['reviews']]
    })
    agent3_res = call_gpt_api(
        system_prompt=PROMPT_REBUTTAL_ANALYZER,
        question=f"Analyze the rebuttal interactions: {rebuttal_input}",
        base64_images=[],  # no images
        model=MODEL_NAME
    )

    # ==========================================
    # AGENT 4: DECISION COORDINATOR
    # ==========================================
    final_input = {
        "raw_scores": [r['score'] for r in processed_data['reviews']],
        "visual_content_eval": agent1_res,  # includes visual evaluation output
        "review_analysis": agent2_res,
        "rebuttal_analysis": agent3_res
    }

    decision_res = call_gpt_api(
        system_prompt=PROMPT_DECISION_COORDINATOR,
        question=f"Make the final decision based on these agent reports: {json.dumps(final_input)}",
        base64_images=[],  # no images
        model=MODEL_NAME
    )

    decision = parse_decision(decision_res)
    return {
        "title": paper_title,
        "prediction": decision.get("final_decision"),
        "final_score": decision.get("final_score"),
        "reasoning": decision.get("rationale"),
        "decision": decision,
        "final_input": final_input,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def load_completed_results(result_file: str) -> set:
    """Load the set of completed sample identifiers (for skipping)."""
    completed = set()
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Use file_path as the unique identifier (title as fallback)
                    if 'file_path' in data:
                        completed.add(data['file_path'])
                    elif 'title' in data:
                        completed.add(data['title'])  # fallback
                except:
                    continue
    return completed


def save_result(result: Dict, result_file: str):
    """Append a single result to a .jsonl file and flush immediately."""
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()  # write to disk immediately to avoid loss on interruption


def load_completed_results(result_file: str) -> set:
    """Load the set of completed sample identifiers (for skipping)."""
    completed = set()
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Use file_path as the unique identifier (title as fallback)
                    if 'file_path' in data:
                        completed.add(data['file_path'])
                    elif 'title' in data:
                        completed.add(data['title'])  # fallback
                except:
                    continue
    return completed


def save_result(result: Dict, result_file: str):
    """Append a single result to a .jsonl file and flush immediately."""
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()  # write to disk immediately to avoid loss on interruption


import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

lock_map = {
    '3D': threading.Lock(),
    'MLLM': threading.Lock(),
    'RL': threading.Lock()
}


def process_single_task(task_args):
    """
    Single-task worker function for multi-thread execution.
    """
    category, subdir, json_path, pdf_path, result_file = task_args

    try:
        # Read JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)

        # Run multi-agent pipeline (time-consuming)
        result = run_multimodal_pipeline(raw_json, pdf_path)
        result['ground_truth'] = subdir

        # Save result (needs locking)
        lock = lock_map.get(category)
        if lock:
            with lock:
                save_result(result, result_file)

        return True, f"Saved to {result_file}"

    except Exception as e:
        return False, f"Failed to process {json_path}: {str(e)}"


def main():
    # Define output result file paths
    RESULT_FILES = {
        '3D': os.path.join(RESULT_DIR, "result_3d.jsonl"),
        'MLLM': os.path.join(RESULT_DIR, "result_mllm.jsonl"),
        'RL': os.path.join(RESULT_DIR, "result_rl.jsonl")
    }

    # 1. Load completed samples
    completed_sets = {}
    for cat, res_file in RESULT_FILES.items():
        # NOTE: load_completed_results should handle file-not-found gracefully
        if os.path.exists(res_file):
            completed_sets[cat] = load_completed_results(res_file)
        else:
            completed_sets[cat] = set()
        print(f"[INFO] Loaded {len(completed_sets[cat])} completed samples for {cat}.")

    # 2. Collect all tasks to process (only build task list here)
    print("[INFO] Scanning directories to collect tasks...")
    all_tasks = []

    for category in categories:
        category_dir = os.path.join(BASE_DIR, category)
        if not os.path.isdir(category_dir):
            print(f"[WARN] Category directory not found: {category_dir}")
            continue

        subdirs = ['oral', 'spotlight', 'poster', 'reject']
        for subdir in subdirs:
            subdir_path = os.path.join(category_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]

            for json_file in json_files:
                json_path = os.path.join(subdir_path, json_file)
                pdf_file = json_file.replace('.json', '.pdf')
                pdf_path = os.path.join(subdir_path, pdf_file)

                # Check if PDF exists
                if not os.path.exists(pdf_path):
                    # Use tqdm.write for safe printing when a progress bar is active
                    continue

                # Check if already processed
                with open(json_path, 'r', encoding='utf-8') as f:
                    raw_json = json.load(f)
                title = raw_json['title']
                if title in completed_sets[category]:
                    continue

                # Pack arguments for task
                task = (category, subdir, json_path, pdf_path, RESULT_FILES[category])
                all_tasks.append(task)

    print(f"[INFO] Total tasks to process: {len(all_tasks)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_task, task) for task in all_tasks]

        # Show progress with tqdm; as_completed yields futures as they finish
        pbar = tqdm(as_completed(futures), total=len(all_tasks), unit="file", desc="Processing")

        for future in pbar:
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                # Use tqdm.write so errors do not break the progress bar layout
                tqdm.write(f"[ERROR] {msg}")

    print(f"\n✅ All done! Successfully processed {success_count} new samples.")


if __name__ == "__main__":
    main()
