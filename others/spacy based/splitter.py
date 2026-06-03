import time
import os
import json
from typing import Dict, Any, List
import spacy
from langdetect import detect, DetectorFactory

# Ensure reproducible language detection results
DetectorFactory.seed = 42

SCHEMA_TEMPLATE = {
    "schema_version": "1.0",
    "document_id": "",
    "original_text": "",
    "translated_text_en": "",
    "segments": []
}

class MultilingualSplitter:
    def __init__(self):
        # Load lightweight models for speed (<200ms target)
        # Disabling NER (Named Entity Recognition) drastically improves performance
        self.nlp_en = spacy.load("en_core_web_sm", disable=["ner"])
        self.nlp_de = spacy.load("de_core_news_sm", disable=["ner"])
        
        # Add custom punctuation boundaries to spaCy sentencizer if needed
        # German guillemets (» «) are typically handled well, but we can enforce them
        for nlp in [self.nlp_en, self.nlp_de]:
            # Custom boundary rules can be injected here if native models miss edge cases
            pass

    def _detect_language(self, text: str) -> str:
        """Helper to classify segment language."""
        if not text.strip():
            return "en"
        try:
            lang = detect(text)
            return lang if lang in ["en", "de"] else "en"
        except Exception:
            return "en" # Default fallback

    def process_document(
        self,
        text: str,
        primary_lang: str = "en",
        document_id: str = "",
        schema_output_path: str = ""
    ) -> Dict[str, Any]:
        """
        Splits text into sentences based on language structures and outputs 
        the validated Task 340 Schema.
        """
        start_time = time.time()
        
        # Select base pipeline depending on overarching/dominant language layout
        nlp = self.nlp_de if primary_lang == "de" else self.nlp_en
        doc = nlp(text)
        
        segments = []
        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Determine fine-grained language per segment
            segment_lang = self._detect_language(sent_text)
            
            segments.append({
                "segment_id": idx,
                "text": sent.text,  # Preserves original spacing inside the segment
                "lang": segment_lang,
                "start_char": sent.start_char,
                "end_char": sent.end_char
            })
            
        # Placeholder translation logic for R&D phase (Task 340 requirement)
        # In production, route 'de' segments to a translation model (e.g., MarianMT or DeepL)
        translated_chunks = []
        for seg in segments:
            if seg["lang"] == "de":
                translated_chunks.append(f"[Translated: {seg['text']}]")
            else:
                translated_chunks.append(seg["text"])

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
                 
        execution_time_ms = (time.time() - start_time) * 1000
        
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

# Quick manual validation
if __name__ == "__main__":
    splitter = MultilingualSplitter()
    sample_mixed_text = 'The system setup is complete. »Das Hauptquartier hat die Datenübertragung gestartet.« Please verify the payload data.'
    
    result = splitter.process_document(sample_mixed_text)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
