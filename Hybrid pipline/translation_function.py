from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import torch
import time
import database_connection

DEFAULT_BATCH_SIZE = 8

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

MODEL_CACHE = {}


def get_translation_resources(source_lang):

    model_name = LANG_MAPPING.get(source_lang)

    if source_lang == "en":
        return None, None

    if not model_name:
        raise ValueError(
            f"Unsupported language detected: {source_lang}"
        )

    if model_name not in MODEL_CACHE:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        MODEL_CACHE[model_name] = (
            tokenizer,
            model.to(device)
        )

    return MODEL_CACHE[model_name]

# ==========================================
# Detect Language
# ==========================================

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"


def normalize_detected_language(source_lang):

    if source_lang in LANG_MAPPING:
        return source_lang

    print(
        f"Unsupported detected language '{source_lang}', "
        "falling back to English passthrough"
    )
    return "en"

# ==========================================
# Translate One Chunk
# ==========================================

def translate_chunk(text, source_lang):

    chunk_start_time = time.perf_counter()

    if source_lang == "en":
        chunk_duration = time.perf_counter() - chunk_start_time
        print(f"Chunk translation completed in {chunk_duration:.2f} seconds")
        return text

    tokenizer, model = get_translation_resources(source_lang)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    translated_tokens = model.generate(
        **inputs,
        max_length=512
    )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    chunk_duration = time.perf_counter() - chunk_start_time
    print(f"Chunk translation completed in {chunk_duration:.2f} seconds")

    return translated_text


def translate_chunks(chunks, source_lang):

    batch_start_time = time.perf_counter()

    if source_lang == "en":
        batch_duration = time.perf_counter() - batch_start_time
        print(f"Batch translation completed in {batch_duration:.2f} seconds")
        return chunks

    tokenizer, model = get_translation_resources(source_lang)

    inputs = tokenizer(
        chunks,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    translated_tokens = model.generate(
        **inputs,
        max_length=512
    )

    translated_chunks = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )

    batch_duration = time.perf_counter() - batch_start_time
    print(f"Batch translation completed in {batch_duration:.2f} seconds")

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


def translate_chunks_in_batches(chunks, source_lang, batch_size=DEFAULT_BATCH_SIZE):

    if source_lang == "en":
        return chunks

    translated_chunks = []
    effective_batch_size = min(batch_size, len(chunks)) if chunks else batch_size
    chunk_batches = list(batch_items(chunks, effective_batch_size))

    print(
        f"Source language: {source_lang} | Chunk count: {len(chunks)} | "
        f"Effective batch size: {effective_batch_size} | "
        f"Batch count: {len(chunk_batches)}"
    )

    for batch_number, chunk_batch in enumerate(chunk_batches, start=1):
        print(
            f"Processing chunk batch {batch_number}/{len(chunk_batches)} "
            f"with {len(chunk_batch)} chunk(s)"
        )
        translated_chunks.extend(translate_chunks(chunk_batch, source_lang))

    return translated_chunks

# ==========================================
# Full Translation Pipeline
# ==========================================

def translate_document_to_english(text):

    document_start_time = time.perf_counter()

    source_lang = "de"

    print(f"Using source language: {source_lang}")

    chunks = chunk_text(text)

    for idx, _ in enumerate(chunks):
        print(
            f"Queued chunk {idx + 1}/{len(chunks)} for batch translation"
        )

    translated_chunks = translate_chunks_in_batches(
        chunks,
        source_lang
    )

    translated_document = "\n".join(translated_chunks)
    total_duration = time.perf_counter() - document_start_time

    print(f"Total translation completed in {total_duration:.2f} seconds")

    return translated_document


def translate_documents_to_english(texts):

    documents_start_time = time.perf_counter()
    translated_documents = []

    if not texts:
        total_duration = time.perf_counter() - documents_start_time
        print(f"Batch document processing completed in {total_duration:.2f} seconds")
        return translated_documents

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

    translated_texts = [None] * len(texts)

    for lang, grouped_items in language_groups.items():
        print(
            f"Processing {len(grouped_items)} document(s) "
            f"for detected language: {lang}"
        )

        grouped_texts = [item["text"] for item in grouped_items]

        if lang == "en":
            translated_group = grouped_texts
        else:
            translated_group = translate_chunks_in_batches(
                grouped_texts,
                lang
            )

        for item, translated_text in zip(grouped_items, translated_group):
            translated_texts[item["original_index"]] = translated_text

    translated_documents = [
        {
            "original_text": original_text,
            "translated_text": translated_texts[idx]
        }
        for idx, original_text in enumerate(texts)
    ]

    total_duration = time.perf_counter() - documents_start_time
    print(f"Batch document processing completed in {total_duration:.2f} seconds")

    return translated_documents