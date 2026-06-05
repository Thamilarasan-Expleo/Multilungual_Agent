from transformers import MarianMTModel, MarianTokenizer
import torch
import time
import re
from markdown import markdown
from bs4 import BeautifulSoup
 
# ==========================================
# Text Cleaning Function
# ==========================================

def clean_text(text: str) -> str:
    """
    Removes all non-alphanumeric characters except basic punctuation and whitespace.
    Keeps: a-z, A-Z, 0-9, whitespace, basic punctuation (.,!?;:()-[]{}'"/@)
    """
    # Remove all non-word characters except specified punctuation
    return re.sub(r'[^\w\s.,!?;:\-\[\]\{\}\'\"/@()]', '', text)

# ==========================================
# Load Model
# ==========================================
 
device = "cuda" if torch.cuda.is_available() else "cpu"
 
# ==========================================
# Language Mapping
# langdetect -> MarianMT model
# ==========================================
 
LANG_MAPPING = {
    "en": None,
    "de": "Helsinki-NLP/opus-mt-de-en",
}

DEFAULT_BATCH_SIZE = 8
MODEL_CACHE = {}

def markdown_to_text(md_text: str) -> str:
    if not md_text:
        return md_text
    html = markdown(md_text)
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
 
 
def get_translation_resources(source_lang):
    load_start = time.perf_counter()
    model_name = LANG_MAPPING.get(source_lang)
 
    if source_lang == "en":
        return None, None
 
    if not model_name:
        raise ValueError(
            f"Unsupported language detected: {source_lang}"
        )
 
    if model_name not in MODEL_CACHE:
        print(f"Loading translation model '{model_name}' on {device}...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        MODEL_CACHE[model_name] = (
            tokenizer,
            model
        )
        load_duration = time.perf_counter() - load_start
        print(f"Translation model loaded in {load_duration:.2f} seconds")
    else:
        load_duration = time.perf_counter() - load_start
        print(f"Translation model cache hit in {load_duration:.4f} seconds")
 
    return MODEL_CACHE[model_name]
 
def translate_chunks(chunks, source_lang, batch_size=DEFAULT_BATCH_SIZE):
    batch_start_time = time.perf_counter()

    total_words = sum(len(text.split()) for text in chunks)
    print(f"Translating {len(chunks)} chunk(s) on {device} | Total words: {total_words}")

    if not chunks:
        print("No chunks to translate")
        return []
 
    if source_lang == "en":
        batch_duration = time.perf_counter() - batch_start_time
        print(f"Batch translation completed in {batch_duration:.2f} seconds")
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
            max_length=512
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                max_length=512
            )

        translated_batch = tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )
        translated_chunks.extend(translated_batch)
        batch_duration = time.perf_counter() - batch_start
        print(f"Batch {batch_idx}/{total_batches} completed in {batch_duration:.2f} seconds")
 
    batch_duration = time.perf_counter() - batch_start_time
    print(f"Batch translation completed in {batch_duration:.2f} seconds")
    print(f"Translation service total time: {batch_duration:.2f} seconds")
 
    return translated_chunks
 
# ==========================================
# Split Large Documents
# ==========================================
 
def chunk_text(text, chunk_size=400):
 
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
        print(f"Batch document processing completed in {total_duration:.2f} seconds")
        return translated_documents
 
    texts = [clean_text(markdown_to_text(text)) for text in texts]

    source_lang = "de"
    print(f"Using source language for all documents: {source_lang}")
 
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
        print(
            f"Processing {len(grouped_items)} document(s) "
            f"for detected language: {lang}"
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
    print(f"Batch document processing completed in {total_duration:.2f} seconds")
 
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
    print("Original German Text:\n",german_text)
    
    result = translate_documents_to_english(german_text)

    print("\nTranslated Output:\n")
    print("length of Translated Output:", len(result))
    print(result)
 
 
