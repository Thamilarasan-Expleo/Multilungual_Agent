import time
import os
import re
import json
from typing import Dict, Any, List
import spacy
import fasttext

SCHEMA_TEMPLATE = {
    "schema_version": "1.0",
    "document_id": "",
    "original_text": "",
    "translated_text_en": "",
    "segments": []
}

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

    def process_document(self, text: str, document_id: str = "", schema_output_path: str = "") -> Dict[str, Any]:
        start_time = time.perf_counter()
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
            
        translated_chunks = []
        for seg in segments:
            if seg["lang"] == "de":
                translated_chunks.append(f"[Translated to EN: {seg['text'].strip()}]")
            else:
                translated_chunks.append(seg["text"].strip())

        schema_segments = [
            {
                "segment_id": seg["segment_id"],
                "text": seg["text"],
                "language": seg["lang"],
                "start_char": seg["start_char"],
                "end_char": seg["end_char"]
            }
            for seg in segments
        ]
                
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        schema_payload = {
            "schema_version": SCHEMA_TEMPLATE["schema_version"],
            "document_id": document_id,
            "original_text": text,
            "translated_text_en": " ".join(translated_chunks),
            "segments": schema_segments
        }
        schema_path = schema_output_path
        if not schema_path:
            if document_id:
                safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", document_id)
                schema_path = os.path.join(os.path.dirname(__file__), f"schema_{safe_id}.json")
            else:
                schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
        with open(schema_path, "w", encoding="utf-8") as schema_file:
            json.dump(schema_payload, schema_file, ensure_ascii=False, indent=2)

        return {
            "schema_version": "1.0.0",
            "original_text": text,
            "translated_en_text": " ".join(translated_chunks),
            "segments": segments,
            "schema": schema_payload,
            "debug_execution_time_ms": execution_time_ms
        }
