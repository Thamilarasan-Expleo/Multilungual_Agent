import time
import os
import re
import json
import uuid
from markdown import markdown
from bs4 import BeautifulSoup
from typing import Dict, Any, List
import spacy
import fasttext
import translation_service

SCHEMA_TEMPLATE = {
    "schema_version": "1.0",
    "document_id": "",
    "original_text": "",
    "translated_text_en": "",
    "translated_text_en_file": "",
    "segments": []
}

_TRANSLATION_RESOURCES = translation_service.get_translation_resources("de")

def markdown_to_text(md_text: str) -> str:
    if not md_text:
        return md_text
    html = markdown(md_text)
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

class MultilingualSplitter:
    def __init__(self, model_path: str = "lid.176.ftz", fasttext_min_confidence: float = 0.20):
        self.nlp = spacy.blank("de") 
        self.nlp.add_pipe("sentencizer")
        self.fasttext_min_confidence = fasttext_min_confidence
        
        self.german_signal_regex = re.compile(
            r'\b(der|die|das|und|ist|mit|von|für|dass|ein|eine|zum|zur|nicht|sind|werden|wurde|wird|auf|um|am|im|es|dem|den|des|zu|vor|nach|oder|wie)\b'
            r'|[äöüßÄÖÜ]'
            r'|\b[A-Z][a-z]+(ung|heit|keit|schaft|tät|ment|tion|richtlinie|befund|aufbau|fehler)\b',
            re.IGNORECASE
        )

        if os.path.exists(model_path):
            self.lang_model = fasttext.load_model(model_path)
            print(f"Loaded FastText language detection model from '{model_path}'")
        else:
            self.lang_model = None
            print(f"Warning: FastText model '{model_path}' not found. Language detection will rely on heuristics only.")

    def _detect_language(self, text: str) -> (str, str):
        cleaned_text = text.strip()
        if not cleaned_text:
            return "en", "regex"
        fasttext_confidence = None
        if self.lang_model:
            cleaned_inline = cleaned_text.replace("\n", " ")
            predictions = self.lang_model.predict(cleaned_inline, k=1)
            lang_tag = predictions[0][0].replace("__label__", "")
            fasttext_confidence = predictions[1][0] if predictions and predictions[1] else 0.0
            if lang_tag in ["en", "de"] and fasttext_confidence >= self.fasttext_min_confidence:
                return lang_tag, f"fasttext/{fasttext_confidence:.4f}"
        confidence_note = "regex"
        if fasttext_confidence is not None:
            confidence_note = f"regex/fasttext_{fasttext_confidence:.4f}"
        return ("de", confidence_note) if self.german_signal_regex.search(cleaned_text) else ("en", confidence_note)

    def translate_segments_to_en(self, de_segments: List[Dict[str, Any]]) -> List[str]:
        if not de_segments:
            return []
        texts = [seg["text"].strip() for seg in de_segments]
        total_words = sum(len(text.split()) for text in texts)
        print(f"Total German segments: {len(texts)} | Total words: {total_words}")
        if texts:
            print(f"Sample DE segment: {texts[0][:160]}")
        translation_start = time.perf_counter()
        translated = translation_service.translate_chunks(texts, "de")
        translation_duration = time.perf_counter() - translation_start
        print(f"German segments translation completed in {translation_duration:.2f} seconds")
        if translated:
            print(f"Sample EN translation: {translated[0][:160]}")
        return translated

    def process_document(self, text: str, document_id: str = "", schema_output_path: str = "", translate: bool = True) -> Dict[str, Any]:
        start_time = time.perf_counter()
        text = markdown_to_text(text)
        doc = self.nlp(text)
        
        segments = []
        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            segment_lang, confidence = self._detect_language(sent_text)
            segments.append({
                "segment_id": idx,
                "text": sent.text,
                "lang": segment_lang,
                "lang_confidence": confidence,
                "start_char": sent.start_char,
                "end_char": sent.end_char
            })

        if translate:
            de_segments = [seg for seg in segments if seg["lang"] == "de"]
            translated_de_chunks = self.translate_segments_to_en(de_segments)
            translated_de_idx = 0

            translated_chunks = []
            for seg in segments:
                if seg["lang"] == "de":
                    translated_text = translated_de_chunks[translated_de_idx]
                    translated_chunks.append(translated_text)
                    seg["translated_text_en"] = translated_text
                    translated_de_idx += 1
                else:
                    translated_text = seg["text"].strip()
                    translated_chunks.append(translated_text)
                    seg["translated_text_en"] = translated_text
        else:
            translated_chunks = []
            for seg in segments:
                translated_text = seg["text"].strip()
                translated_chunks.append(translated_text)
                seg["translated_text_en"] = translated_text

        translated_document_text = "\n\n".join(
            chunk.strip() for chunk in translated_chunks if chunk.strip()
        )

        schema_segments = [
            {
                "segment_id": seg["segment_id"],
                "text": seg["text"],
                "language": seg["lang"],
                "translated_text_en": seg["translated_text_en"],
                "start_char": seg["start_char"],
                "end_char": seg["end_char"]
            }
            for seg in segments
        ]
                
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        schema_path = schema_output_path
        if not schema_path:
            if document_id:
                safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", document_id)
                schema_path = os.path.join(os.path.dirname(__file__), f"schema_{safe_id}.json")
            else:
                schema_path = os.path.join(os.path.dirname(__file__), "schema.json")

        schema_dir = os.path.dirname(schema_path) or os.path.dirname(__file__)
        translated_markdown_file = f"{uuid.uuid4()}.md"
        translated_markdown_path = os.path.join(schema_dir, translated_markdown_file)

        with open(translated_markdown_path, "w", encoding="utf-8") as translated_file:
            translated_file.write(translated_document_text)

        with open(schema_path, "w", encoding="utf-8") as schema_file:
            schema_payload = {
                "schema_version": SCHEMA_TEMPLATE["schema_version"],
                "document_id": document_id,
                "original_text": text,
                "translated_text_en": translated_document_text,
                "translated_text_en_file": translated_markdown_file,
                "segments": schema_segments
            }
            json.dump(schema_payload, schema_file, ensure_ascii=False, indent=2)

        return {
            "schema_version": "1.0.0",
            "original_text": text,
            "translated_en_text": translated_document_text,
            "segments": segments,
            "schema": schema_payload,
            "debug_execution_time_ms": execution_time_ms
        }
