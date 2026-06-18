"""
PDF Question Extractor (LLM Vision + PyMuPDF Images)
=====================================================
Uses an LLM (OpenRouter or Gemini) to extract questions & options from PDF
pages, and PyMuPDF to extract diagram images. Merges both into a JSON file.

All math/scientific notation is returned in LaTeX format ($...$) for
rendering with KaTeX / react-katex.

Setup:
    pip install PyMuPDF requests

Usage:
    python extract_questions.py exam.pdf --api-key YOUR_OPENROUTER_KEY
    python extract_questions.py exam.pdf --provider gemini --api-key YOUR_GEMINI_KEY
"""

import fitz  # PyMuPDF
import os
import json
import argparse
import sys
import base64
import time
import requests


# ---------------------------------------------------------------------------
# Image extraction (kept from existing PyMuPDF pipeline)
# ---------------------------------------------------------------------------

def extract_images_from_page(page: fitz.Page, image_dir: str,
                             page_number: int) -> list:
    """
    Extract diagram/figure images from a PDF page using PyMuPDF.
    Returns a list of dicts with image metadata and file paths.
    Filters out noise (tiny icons, margin logos/watermarks).
    """
    page_dict = page.get_text("dict", sort=True)
    page_height = page_dict.get("height", page.rect.height)

    images = []
    image_counter = 0

    for block in page_dict["blocks"]:
        if block["type"] != 1:  # only image blocks
            continue

        y_top = block["bbox"][1]
        image_counter += 1
        width = block.get("width", 0)
        height = block.get("height", 0)

        # Skip tiny images (bullets, icons, artifacts)
        if width < 20 or height < 20:
            continue

        # Skip images in the very bottom margin (watermarks/logos)
        if y_top > page_height - (page_height * 0.08):
            continue

        ext = block.get("ext", "png")
        if ext not in ("png", "jpg", "jpeg", "bmp", "tiff"):
            ext = "png"
        filename = f"page{page_number + 1}_diagram_{image_counter}.{ext}"
        filepath = os.path.join(image_dir, filename)

        image_bytes = block.get("image")
        if image_bytes:
            with open(filepath, "wb") as f:
                f.write(image_bytes)

            images.append({
                "y": y_top,
                "path": filepath,
                "relative_path": os.path.join(
                    os.path.basename(image_dir), filename
                ),
            })

    return images


# ---------------------------------------------------------------------------
# Render PDF page to PNG for LLM vision
# ---------------------------------------------------------------------------

def render_page_to_png(page: fitz.Page, dpi: int = 200) -> bytes:
    """Render a PDF page to PNG bytes at the given DPI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = r"""You are an expert at reading exam question papers.

Look at this PDF page image and extract ALL questions and their multiple-choice options.

Rules:
- Extract EVERY question visible on the page, do not skip any.
- For each question, extract the question number, the full question text, and all options.
- **CRITICAL: All mathematical expressions, scientific symbols, Greek letters, fractions,
  superscripts, subscripts, summations, integrals, square roots, vectors, matrices,
  chemical formulas, and any special notation MUST be written in LaTeX format.**
  Wrap each LaTeX expression with dollar signs: $...$  for inline math.
  Examples:
    - Summation: $\sum_{i=1}^{n} x_i$
    - Fraction: $\frac{a}{b}$
    - Greek letters: $\alpha$, $\beta$, $\mu$, $\nu$
    - Superscript/subscript: $x^2$, $a_n$, $m/s^2$
    - Square root: $\sqrt{5}$
    - Vectors: $\vec{F}$
    - Integrals: $\int_0^1 f(x)\,dx$
    - Chemical: $H_2O$, $CO_2$
    - Combined: $\frac{\sum_{i=1}^{n} x_i}{n}$
  Even simple things like "m/s2" should become $m/s^2$.
- Ignore page headers, footers, page numbers, watermarks, and any non-question content.
- If a question has a diagram/figure, note it in the question text as "[See diagram]" but do NOT describe the diagram.
- Options may be labeled with letters (A, B, C, D) or numbers (1, 2, 3, 4).
- Return ONLY valid JSON, no extra text.

Return the data as a JSON array where each element has:
{
  "question_number": "string (e.g. 'Q.6', '1', 'Q1')",
  "question_text": "string (full question text with LaTeX math)",
  "options": {
    "A": "option text with LaTeX math",
    "B": "option text with LaTeX math",
    ...
  }
}

If the page contains no questions, return an empty array: []
"""


# ---------------------------------------------------------------------------
# Provider: OpenRouter (default, works with free models)
# ---------------------------------------------------------------------------

def extract_with_openrouter(api_key: str, page_png: bytes,
                            model: str = "google/gemma-4-26b-a4b-it:free") -> list:
    """
    Send a rendered PDF page image to OpenRouter and get back
    structured question data. Retries on rate limits with fallback models.
    """
    b64_image = base64.b64encode(page_png).decode("utf-8")

    # Fallback free models if primary is rate-limited
    FREE_MODELS = [
        model,
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "nex-agi/nex-n2-pro:free",
    ]
    # Remove duplicates while preserving order
    seen = set()
    models_to_try = []
    for m in FREE_MODELS:
        if m not in seen:
            seen.add(m)
            models_to_try.append(m)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt_model in models_to_try:
        for attempt in range(3):
            payload = {
                "model": attempt_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                },
                            },
                            {
                                "type": "text",
                                "text": EXTRACTION_PROMPT,
                            },
                        ],
                    }
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }

            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )

            if resp.status_code == 200:
                data = resp.json()
                try:
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return parsed
                    if isinstance(parsed, dict) and "questions" in parsed:
                        return parsed["questions"]
                    return []
                except (json.JSONDecodeError, TypeError, KeyError):
                    print(f"\n  WARNING: Could not parse response as JSON.")
                    return []

            elif resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"\n  Rate limited ({attempt_model}). "
                      f"Retrying in {wait}s...", end="", flush=True)
                time.sleep(wait)
            else:
                print(f"\n  ERROR [{resp.status_code}]: {resp.text[:200]}")
                break  # Non-retryable error, try next model

        # If all retries exhausted for this model, try next one

    print(f"\n  All models exhausted. Could not extract questions.")
    return []


# ---------------------------------------------------------------------------
# Provider: Gemini (direct API)
# ---------------------------------------------------------------------------

def extract_with_gemini(api_key: str, page_png: bytes,
                        model: str = "gemini-2.0-flash") -> list:
    """
    Send a rendered PDF page image to Gemini API directly via REST.
    No SDK required.
    """
    b64_image = base64.b64encode(page_png).decode("utf-8")

    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{model}:generateContent?key={api_key}")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": b64_image,
                        }
                    },
                    {
                        "text": EXTRACTION_PROMPT,
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    resp = requests.post(url, json=payload, timeout=120)

    if resp.status_code != 200:
        error = resp.json().get("error", {})
        msg = error.get("message", resp.text[:200])
        print(f"\n  ERROR [{resp.status_code}]: {msg}")
        return []

    data = resp.json()
    try:
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "questions" in parsed:
            return parsed["questions"]
        return []
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        print(f"\n  WARNING: Could not parse Gemini response as JSON.")
        return []


# ---------------------------------------------------------------------------
# Merge LLM questions with PyMuPDF images
# ---------------------------------------------------------------------------

def merge_questions_with_images(questions: list, images: list) -> list:
    """
    Attach extracted diagram paths to questions.
    Each image is attached to the question that appears just before it
    in page order (by question index).
    """
    if not images or not questions:
        for q in questions:
            if "diagram" not in q:
                q["diagram"] = None
        return questions

    num_q = len(questions)
    for i, img in enumerate(images):
        if i < num_q:
            questions[i]["diagram"] = img["relative_path"]

    for q in questions:
        if "diagram" not in q:
            q["diagram"] = None

    return questions


# ---------------------------------------------------------------------------
# Page range parser
# ---------------------------------------------------------------------------

def parse_page_range(pages_str: str, total_pages: int) -> list:
    if not pages_str or pages_str.lower() == "all":
        return list(range(total_pages))

    page_set = set()
    parts = pages_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = max(1, int(start.strip()))
            end = min(total_pages, int(end.strip()))
            for p in range(start, end + 1):
                page_set.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= total_pages:
                page_set.add(p - 1)
    return sorted(page_set)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def extract_from_pdf(pdf_path: str, api_key: str,
                     provider: str = "gemini",
                     pages_str: str = "all",
                     output_json: str = "output.json",
                     image_dir: str = "extracted_diagrams",
                     model: str = None):
    if not os.path.isfile(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    os.makedirs(image_dir, exist_ok=True)

    # Set default model per provider
    if model is None:
        if provider == "gemini":
            model = "gemini-2.0-flash"
        else:
            # google/gemini-2.5-flash-lite is cheapest paid vision model
            # Use "google/gemma-4-26b-a4b-it:free" for fully free (lower quality)
            model = "google/gemini-2.5-flash-lite"

    # Pick the right extraction function
    if provider == "gemini":
        extract_fn = lambda png: extract_with_gemini(api_key, png, model)
    else:
        extract_fn = lambda png: extract_with_openrouter(api_key, png, model)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Opened '{pdf_path}' -- {total_pages} page(s) total.")
    print(f"Provider: {provider} | Model: {model}\n")

    page_indices = parse_page_range(pages_str, total_pages)
    if not page_indices:
        print("ERROR: No valid pages to process.")
        sys.exit(1)

    all_questions = []

    for page_idx in page_indices:
        page = doc[page_idx]
        print(f"  Page {page_idx + 1}/{total_pages} ...", end=" ", flush=True)

        # Step 1: Extract images using PyMuPDF (existing pipeline)
        images = extract_images_from_page(page, image_dir, page_idx)

        # Step 2: Render page to PNG and send to LLM
        page_png = render_page_to_png(page)
        questions = extract_fn(page_png)

        # Step 3: Merge images into questions
        questions = merge_questions_with_images(questions, images)

        print(f"found {len(questions)} question(s), "
              f"{len(images)} diagram(s)")
        all_questions.extend(questions)

        # Small delay to respect rate limits
        if page_idx != page_indices[-1]:
            time.sleep(1)

    # Write JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_questions)} question(s) -> {output_json}")
    print(f"Diagrams saved in -> {image_dir}/")

    doc.close()
    return all_questions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MCQ questions from PDF using LLM vision + PyMuPDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using OpenRouter (default, recommended):
  python extract_questions.py exam.pdf --api-key sk-or-...
  
  # Using Gemini directly:
  python extract_questions.py exam.pdf --provider gemini --api-key AIza...

  # Process specific pages:
  python extract_questions.py exam.pdf --pages 1-5 --api-key sk-or-...

Get your free API key:
  OpenRouter: https://openrouter.ai/keys
  Gemini:     https://aistudio.google.com/apikey
        """,
    )
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument(
        "--pages", default="all",
        help="Pages to process (e.g. 'all', '1', '1-5', '1,3,7')",
    )
    parser.add_argument(
        "--output", default="output.json",
        help="Output JSON file path (default: output.json)",
    )
    parser.add_argument(
        "--image-dir", default="extracted_diagrams",
        help="Directory for extracted diagrams (default: extracted_diagrams/)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (or set OPENROUTER_API_KEY / GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--provider", default="gemini", choices=["gemini", "openrouter"],
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (default: auto per provider)",
    )
    args = parser.parse_args()

    # Resolve API key
    if args.api_key:
        api_key = args.api_key
    elif args.provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        print("ERROR: No API key provided.")
        if args.provider == "openrouter":
            print("Set via --api-key or OPENROUTER_API_KEY env var.")
            print("Get a free key at: https://openrouter.ai/keys")
        else:
            print("Set via --api-key or GEMINI_API_KEY env var.")
            print("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    extract_from_pdf(
        args.pdf, api_key, args.provider, args.pages,
        args.output, args.image_dir, args.model,
    )
