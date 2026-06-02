import time
import re
import os
import json
from typing import Dict, Any, List
import spacy

SCHEMA_TEMPLATE = {
    "schema_version": "1.0",
    "document_id": "",
    "original_text": "",
    "translated_text_en": "",
    "segments": []
}

class MultilingualSplitter:
    def __init__(self):
        # PERFORMANCE OPTIMIZATION: Blank rule-based components bypass heavy network graphs
        self.nlp_en = spacy.blank("en")
        self.nlp_en.add_pipe("sentencizer")
        
        self.nlp_de = spacy.blank("de")
        self.nlp_de.add_pipe("sentencizer")
        
        # Compiled expressions targeting unique high-frequency language distributions
        self.de_regex = re.compile(
            r'\b(der|die|das|und|ist|mit|von|für|dass|ein|eine|zum|zur|nicht|sind|werden|wurde|wird)\b|[äöüßÄÖÜ]|\b[A-Z][a-z]+(ung|heit|keit|schaft|tät|ment|tion)\b', 
            re.IGNORECASE
        )

    def _detect_language(self, text: str) -> str:
        """Heuristic language engine running in sub-millisecond ranges."""
        if not text.strip():
            return "en"
        
        de_matches = len(self.de_regex.findall(text))
        return "de" if de_matches > 0 else "en"

    def process_document(
        self,
        text: str,
        primary_lang: str = "en",
        document_id: str = "",
        schema_output_path: str = ""
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        nlp = self.nlp_de if primary_lang == "de" else self.nlp_en
        doc = nlp(text)
        
        segments = []
        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            segment_lang = self._detect_language(sent_text)
            segments.append({
                "segment_id": idx,
                "text": sent.text,
                "lang": segment_lang,
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
