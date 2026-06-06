import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from transformers import MarianMTModel, MarianTokenizer
import torch
import time
import re
import logging
from markdown import markdown
from bs4 import BeautifulSoup
from dotenv import load_dotenv
 
# ==========================================
# Text Cleaning Function
# ==========================================

def clean_text(text: str) -> str:
    """
    Removes all non-alphanumeric characters except basic punctuation and whitespace.
    Keeps: a-z, A-Z, 0-9, whitespace, basic punctuation (.,!?;:()-[]{}'"/@)
    """
    logger.debug("Cleaning text for translation")
    # Remove all non-word characters except specified punctuation
    return re.sub(r'[^\w\s.,!?;:\-\[\]\{\}\'\"/@()]', '', text)

# ==========================================
# Load Model
# ==========================================
 
load_dotenv()

LOG_LEVEL = os.getenv("TRANSLATION_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Translation device selected: %s", device)
 
# ==========================================
# Language Mapping
# langdetect -> MarianMT model
# ==========================================
 
LANG_MAPPING = {
    "en": None,
    "de": os.getenv("TRANSLATION_MODEL_DE", "Helsinki-NLP/opus-mt-de-en"),
}

DEFAULT_BATCH_SIZE = int(os.getenv("TRANSLATION_BATCH_SIZE", "8"))
MODEL_CACHE = {}

def markdown_to_text(md_text: str) -> str:
    if not md_text:
        logger.debug("markdown_to_text called with empty text")
        return md_text
    logger.debug("markdown_to_text converting markdown to text")
    html = markdown(md_text)
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
 
 
def get_translation_resources(source_lang):
    load_start = time.perf_counter()
    model_name = LANG_MAPPING.get(source_lang)
 
    if source_lang == "en":
        logger.info("Source language is English; no translation resources required")
        return None, None
 
    if not model_name:
        raise ValueError(
            f"Unsupported language detected: {source_lang}"
        )

    if os.getenv("TRANSLATION_MODEL_DE") and not os.path.exists(model_name):
        logger.warning(
            "TRANSLATION_MODEL_DE is set but path does not exist: %s",
            model_name,
        )
 
    if model_name not in MODEL_CACHE:
        logger.info("Loading translation model '%s' on %s...", model_name, device)
        tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=True)
        model = MarianMTModel.from_pretrained(model_name, local_files_only=True)
        model = model.to(device)
        model.eval()
        MODEL_CACHE[model_name] = (
            tokenizer,
            model
        )
        load_duration = time.perf_counter() - load_start
        logger.info("Translation model loaded in %.2f seconds", load_duration)
    else:
        load_duration = time.perf_counter() - load_start
        logger.info("Translation model cache hit in %.4f seconds", load_duration)
 
    return MODEL_CACHE[model_name]
 
def translate_chunks(chunks, source_lang, batch_size=DEFAULT_BATCH_SIZE):
    batch_start_time = time.perf_counter()

    total_words = sum(len(text.split()) for text in chunks)
    logger.info("Translating %s chunk(s) on %s | Total words: %s", len(chunks), device, total_words)

    if not chunks:
        logger.info("No chunks to translate")
        return []
 
    if source_lang == "en":
        batch_duration = time.perf_counter() - batch_start_time
        logger.info("Batch translation completed in %.2f seconds", batch_duration)
        return chunks
 
    tokenizer, model = get_translation_resources(source_lang)

    translated_chunks = []
    total_batches = max(1, (len(chunks) + batch_size - 1) // batch_size)

    for batch_idx, batch in enumerate(batch_items(chunks, batch_size), start=1):
        batch = [clean_text(text) for text in batch]
        batch_start = time.perf_counter()
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=int(os.getenv("TRANSLATION_MAX_LENGTH", "512"))
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                max_length=int(os.getenv("TRANSLATION_MAX_LENGTH", "512"))
            )

        translated_batch = tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )
        translated_chunks.extend(translated_batch)
        batch_duration = time.perf_counter() - batch_start
        logger.info("Batch %s/%s completed in %.2f seconds", batch_idx, total_batches, batch_duration)
 
    batch_duration = time.perf_counter() - batch_start_time
    logger.info("Translation service total time: %.2f seconds", batch_duration)
 
    return translated_chunks
 
# ==========================================
# Split Large Documents
# ==========================================
 
def chunk_text(text, chunk_size=None):

    if chunk_size is None:
        chunk_size = int(os.getenv("TRANSLATION_CHUNK_SIZE", "400"))
 
    words = text.split()
 
    chunks = []
 
    for i in range(0, len(words), chunk_size):
        chunks.append(
            " ".join(words[i:i + chunk_size])
        )
 
    return chunks
 
 
def batch_items(items, batch_size):
 
    for idx in range(0, len(items), batch_size):
        yield items[idx:idx + batch_size]
 
# ==========================================
# Full Translation Pipeline
# ==========================================
 
 
def translate_documents_to_english(texts):

    documents_start_time = time.perf_counter()
    translated_documents = []
 
    if not texts:
        total_duration = time.perf_counter() - documents_start_time
        logger.info("Batch document processing completed in %.2f seconds", total_duration)
        return translated_documents
 
    texts = [clean_text(markdown_to_text(text)) for text in texts]

    source_lang = "de"
    logger.info("Using source language for all documents: %s", source_lang)
 
    language_groups = {
        source_lang: [
            {
                "original_index": idx,
                "text": text
            }
            for idx, text in enumerate(texts)
        ]
    }
 
    translated_documents = [None] * len(texts)
 
    for lang, grouped_items in language_groups.items():
        logger.info(
            "Processing %s document(s) for detected language: %s",
            len(grouped_items),
            lang,
        )
 
        grouped_texts = [item["text"] for item in grouped_items]
 
        if lang == "en":
            translated_group = grouped_texts
        else:
            translated_group = translate_chunks(
                grouped_texts,
                lang,
                batch_size=DEFAULT_BATCH_SIZE
            )
 
        for item, translated_text in zip(grouped_items, translated_group):
            translated_documents[item["original_index"]] = translated_text
 
    total_duration = time.perf_counter() - documents_start_time
    logger.info("Batch document processing completed in %.2f seconds", total_duration)
 
    return translated_documents
 
# ==========================================
# Example
# ==========================================
 
if __name__ == "__main__":
    import json
    import os

    schema_path = os.path.join(
        os.path.dirname(__file__),
        "schema_TEST_DATA.json"
    )

    with open(schema_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    segments = payload.get("segments", [])
    german_text = [
        markdown_to_text(segment["text"])
        for segment in segments
        if segment.get("language") == "de"
    ]
    logger.info("Original German Text length: %s", len(german_text))
    
    result = translate_documents_to_english(german_text)

    logger.info("Translated Output length: %s", len(result))
 
 
