
import time
import os
import re
import json
import uuid
import logging
from markdown import markdown
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import spacy
import fasttext
import translation_service
from db_connection import fetch_translation_cache, insert_translation_cache

SCHEMA_TEMPLATE = {
    "schema_version": "1.0",
    "document_id": "",
    "original_text": "",
    "translated_text_en": "",
    "translated_text_en_file": "",
    "segments": []
}

load_dotenv()

LOG_LEVEL = os.getenv("SPLITTER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.getenv("SPLITTER_MODEL_PATH", "lid.176.ftz")
DEFAULT_FASTTEXT_MIN_CONFIDENCE = float(
    os.getenv("SPLITTER_FASTTEXT_MIN_CONFIDENCE", "0.20")
)
DEFAULT_SOURCE_LANG = os.getenv("SPLITTER_SOURCE_LANG", "de")
MOUNTED_FOLDER = os.getenv("MOUNTED_FOLDER", "translated_data")


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


class MultilingualSplitter:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        fasttext_min_confidence: float = DEFAULT_FASTTEXT_MIN_CONFIDENCE,
        source_lang: str = DEFAULT_SOURCE_LANG,
    ):
        logger.info(
            "Initializing MultilingualSplitter with model_path=%s, min_confidence=%.2f, source_lang=%s",
            model_path,
            fasttext_min_confidence,
            source_lang,
        )
        self.nlp = spacy.blank("de")
        self.nlp.add_pipe("sentencizer")
        self.fasttext_min_confidence = fasttext_min_confidence
        self.source_lang = source_lang

        self.german_signal_regex = re.compile(
            r'\b(der|die|das|und|ist|mit|von|für|dass|ein|eine|zum|zur|nicht|sind|werden|wurde|wird|auf|um|am|im|es|dem|den|des|zu|vor|nach|oder|wie)\b'
            r'|[äöüßÄÖÜ]'
            r'|\b[A-Z][a-z]+(ung|heit|keit|schaft|tät|ment|tion|richtlinie|befund|aufbau|fehler)\b',
            re.IGNORECASE,
        )

        if os.path.exists(model_path):
            self.lang_model = fasttext.load_model(model_path)
            logger.info("Loaded FastText language detection model from '%s'", model_path)
        else:
            self.lang_model = None
            logger.warning(
                "FastText model '%s' not found. Language detection will rely on heuristics only.",
                model_path,
            )

    @classmethod
    def preload_resources(
        cls,
        model_path: str = DEFAULT_MODEL_PATH,
        fasttext_min_confidence: float = DEFAULT_FASTTEXT_MIN_CONFIDENCE,
        source_lang: str = DEFAULT_SOURCE_LANG,
    ):
        """Warm all translation resources during service startup."""
        splitter = cls(
            model_path=model_path,
            fasttext_min_confidence=fasttext_min_confidence,
            source_lang=source_lang,
        )
        translation_service.preload_translation_models([source_lang])
        return splitter

    def _detect_language(self, text: str) -> Tuple[str, str]:
        cleaned_text = text.strip()
        if not cleaned_text:
            logger.debug("_detect_language received empty text; defaulting to en")
            return "en", "regex"

        fasttext_confidence = None
        if self.lang_model:
            cleaned_inline = cleaned_text.replace("\n", " ")
            predictions = self.lang_model.predict(cleaned_inline, k=1)
            lang_tag = predictions[0][0].replace("__label__", "")
            fasttext_confidence = predictions[1][0] if predictions and predictions[1] else 0.0
            if lang_tag in ["en", "de"] and fasttext_confidence >= self.fasttext_min_confidence:
                logger.debug(
                    "FastText detected language=%s confidence=%.4f",
                    lang_tag,
                    fasttext_confidence,
                )
                return lang_tag, f"fasttext/{fasttext_confidence:.4f}"

        confidence_note = "regex"
        if fasttext_confidence is not None:
            confidence_note = f"regex/fasttext_{fasttext_confidence:.4f}"
        return (
            "de",
            confidence_note,
        ) if self.german_signal_regex.search(cleaned_text) else ("en", confidence_note)

    def _normalize_text(self, text: str) -> str:
        return markdown_to_text(text)

    def _split_into_segments(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        segments: List[Dict[str, Any]] = []

        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            segment_lang, confidence = self._detect_language(sent_text)
            segments.append(
                {
                    "segment_id": idx,
                    "text": sent.text,
                    "lang": segment_lang,
                    "lang_confidence": confidence,
                    "start_char": sent.start_char,
                    "end_char": sent.end_char,
                }
            )

        return segments

    def _translate_segments(
        self,
        segments: List[Dict[str, Any]],
        translate: bool,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        translated_chunks: List[str] = []

        if not segments:
            return translated_chunks, segments

        if translate:
            de_segments = [seg for seg in segments if seg["lang"] == "de"]
            translated_de_chunks = self.translate_segments_to_en(de_segments)
            translated_de_idx = 0

            for seg in segments:
                if seg["lang"] == "de":
                    translated_text = (
                        translated_de_chunks[translated_de_idx]
                        if translated_de_idx < len(translated_de_chunks)
                        else seg["text"].strip()
                    )
                    translated_de_idx += 1
                else:
                    translated_text = seg["text"].strip()

                seg["translated_text_en"] = translated_text
                translated_chunks.append(translated_text)
        else:
            for seg in segments:
                translated_text = seg["text"].strip()
                seg["translated_text_en"] = translated_text
                translated_chunks.append(translated_text)

        return translated_chunks, segments

    def _reconstruct_translated_text(
        self,
        original_text: str,
        segments: List[Dict[str, Any]],
        translated_chunks: List[str],
    ) -> str:
        if not original_text:
            return ""
        if not segments:
            return original_text

        output_parts = []
        cursor = 0

        for seg, translated in zip(segments, translated_chunks):
            start_char = seg["start_char"]
            end_char = seg["end_char"]

            if cursor < start_char:
                output_parts.append(original_text[cursor:start_char])

            output_parts.append(translated.strip())
            cursor = end_char

        if cursor < len(original_text):
            output_parts.append(original_text[cursor:])

        return "".join(output_parts)

    def _build_processing_bundle(
        self,
        text: str,
        document_id: str = "",
        translate: bool = True,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        normalized_text = self._normalize_text(text)
        segments = self._split_into_segments(normalized_text)
        translated_chunks, segments = self._translate_segments(segments, translate)
        translated_document_text = self._reconstruct_translated_text(
            normalized_text,
            segments,
            translated_chunks,
        )

        language_map = [
            {
                "segment_id": seg["segment_id"],
                "language": seg["lang"],
                "lang_confidence": seg["lang_confidence"],
                "start_char": seg["start_char"],
                "end_char": seg["end_char"],
                "translated_text_en": seg.get("translated_text_en", seg["text"].strip()),
            }
            for seg in segments
        ]

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "document_id": document_id,
            "original_text": normalized_text,
            "segments": segments,
            "translated_chunks": translated_chunks,
            "translated_text_en": translated_document_text,
            "language_map": language_map,
            "debug_execution_time_ms": execution_time_ms,
        }

    def _language_summary(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        de_count = sum(1 for seg in segments if seg.get("lang") == "de")
        en_count = sum(1 for seg in segments if seg.get("lang") == "en")
        total_count = len(segments)

        de_ratio = round(de_count / total_count, 4) if total_count > 0 else 0.0
        en_ratio = round(en_count / total_count, 4) if total_count > 0 else 0.0

        if de_ratio > en_ratio:
            dominant_language = "de"
            adherence_score = de_ratio
        elif en_ratio > de_ratio:
            dominant_language = "en"
            adherence_score = en_ratio
        else:
            dominant_language = "equal"
            adherence_score = de_ratio

        return {
            "language_distribution": {
                "de": de_ratio,
                "en": en_ratio,
            },
            "dominant_language": dominant_language,
            "adherence_score": round(adherence_score, 4),
            "segment_counts": {
                "de": de_count,
                "en": en_count,
                "total": total_count,
            },
        }

    def _build_schema_payload(self, bundle: Dict[str, Any], document_id: str) -> Dict[str, Any]:
        schema_segments = [
            {
                "segment_id": seg["segment_id"],
                "text": seg["text"],
                "language": seg["lang"],
                "translated_text_en": seg.get("translated_text_en", seg["text"].strip()),
                "start_char": seg["start_char"],
                "end_char": seg["end_char"],
            }
            for seg in bundle["segments"]
        ]

        return {
            "schema_version": SCHEMA_TEMPLATE["schema_version"],
            "document_id": document_id,
            "original_text": bundle["original_text"],
            "translated_text_en": bundle["translated_text_en"],
            "translated_text_en_file": "",
            "segments": schema_segments,
        }

    def _build_vector_payload(self, bundle: Dict[str, Any], document_id: str) -> Dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "document_id": document_id,
            "original_text": bundle["original_text"],
            "translated_text_en": bundle["translated_text_en"],
            "language_map": bundle["language_map"],
            "segments": bundle["segments"],
            "debug_execution_time_ms": bundle["debug_execution_time_ms"],
        }

    def translate_segments_to_en(self, de_segments: List[Dict[str, Any]]) -> List[str]:
        if not de_segments:
            logger.info("No German segments to translate")
            return []

        texts = [seg["text"].strip() for seg in de_segments]
        total_words = sum(len(text.split()) for text in texts)
        logger.info("Total German segments: %s | Total words: %s", len(texts), total_words)
        if texts:
            logger.debug("Sample DE segment: %s", texts[0][:160])

        translation_start = time.perf_counter()
        translated = self.cache_translation(texts)
        translation_duration = time.perf_counter() - translation_start

        if translated:
            logger.debug("Sample EN translation: %s", translated[0][:160])
        logger.info("Translation (with cache) completed in %.2f seconds", translation_duration)
        return translated

    def cache_translation(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        cache_hits = fetch_translation_cache(texts)
        to_translate = [text for text in texts if text not in cache_hits]
        logger.info(
            "Cache matches: %s | Total segments: %s",
            len(cache_hits),
            len(texts),
        )

        if to_translate:
            logger.info("Cache miss count: %s", len(to_translate))
            translated_new = translation_service.translate_chunks(to_translate, self.source_lang)
            if translated_new:
                insert_translation_cache(
                    [
                        {
                            "original_text": original,
                            "translated_text": translated,
                        }
                        for original, translated in zip(to_translate, translated_new)
                    ]
                )
                logger.info("Stored %s translated segment(s) to cache", len(translated_new))
            translation_map = {original: translated for original, translated in zip(to_translate, translated_new)}
        else:
            logger.info("Cache hit for all segments")
            translation_map = {}

        combined = {
            **cache_hits,
            **translation_map,
        }

        return [combined.get(text, text) for text in texts]

    def build_vectorization_payload(
        self,
        text: str,
        document_id: str = "",
        translate: bool = True,
    ) -> Dict[str, Any]:
        logger.info(
            "Building vectorization payload id=%s translate=%s",
            document_id or "<none>",
            translate,
        )

        bundle = self._build_processing_bundle(
            text=text,
            document_id=document_id,
            translate=translate,
        )
        return self._build_vector_payload(bundle, document_id)

    def process_document(
        self,
        text: str,
        document_id: str = "",
        schema_output_path: str = "",
        translate: bool = True,
    ) -> Dict[str, Any]:
        logger.info("Processing document id=%s translate=%s", document_id or "<none>", translate)

        bundle = self._build_processing_bundle(
            text=text,
            document_id=document_id,
            translate=translate,
        )

        mounted_root = os.path.join(os.path.dirname(__file__), MOUNTED_FOLDER)
        os.makedirs(mounted_root, exist_ok=True)

        schema_path = schema_output_path
        if not schema_path:
            if document_id:
                safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", document_id)
                schema_path = os.path.join(mounted_root, f"schema_{safe_id}.json")
            else:
                schema_path = os.path.join(mounted_root, "schema.json")

        translated_output_dir = os.path.dirname(schema_path) or mounted_root
        translated_markdown_file = f"{uuid.uuid4()}.md"
        translated_markdown_path = os.path.join(translated_output_dir, translated_markdown_file)

        # The translated markdown file is no longer written because vectorization
        # now consumes the reconstructed translated text directly from payload.
        # with open(translated_markdown_path, "w", encoding="utf-8") as translated_file:
        #     translated_file.write(bundle["translated_text_en"])
        # logger.info("Translated markdown written to %s", translated_markdown_path)

        schema_payload = self._build_schema_payload(bundle, document_id)

        # The schema JSON file is no longer written because the ingest path now
        # consumes the in-memory payload directly.
        # with open(schema_path, "w", encoding="utf-8") as schema_file:
        #     json.dump(schema_payload, schema_file, ensure_ascii=False, indent=2)
        # logger.info("Schema written to %s", schema_path)

        return {
            "schema_version": "1.0.0",
            "original_text": bundle["original_text"],
            "translated_en_text": bundle["translated_text_en"],
            "segments": bundle["segments"],
            "schema": schema_payload,
            "debug_execution_time_ms": bundle["debug_execution_time_ms"],
            "translated_markdown_path": "",
            "schema_path": "",
        }

    def process_text(
        self,
        text: str,
        document_id: str = "",
        schema_output_path: str = "",
        translate: bool = True,
    ) -> Dict[str, Any]:
        bundle = self._build_processing_bundle(
            text=text,
            document_id=document_id,
            translate=translate,
        )

        summary = self._language_summary(bundle["segments"])

        return {
            "translated_text_en": bundle["translated_text_en"],
            "language_distribution": summary["language_distribution"],
            "dominant_language": summary["dominant_language"],
            "adherence_score": summary["adherence_score"],
            "segment_counts": summary["segment_counts"],
            "segments": bundle["segments"],
        }

    def process_markdown_file(
        self,
        markdown_path: str,
        document_id: str = "",
        schema_output_path: str = "",
        translate: bool = True,
    ) -> Dict[str, Any]:
        if not os.path.exists(markdown_path):
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

        with open(markdown_path, "r", encoding="utf-8") as handle:
            raw_text = handle.read()

        if not document_id:
            document_id = os.path.splitext(os.path.basename(markdown_path))[0]

        bundle = self._build_processing_bundle(
            text=raw_text,
            document_id=document_id,
            translate=translate,
        )
        schema_payload = self._build_schema_payload(bundle, document_id)
        vector_payload = self._build_vector_payload(bundle, document_id)

        return {
            "translated_markdown_path": "",
            "schema_path": "",
            "schema": schema_payload,
            "document_id": document_id,
            "debug_execution_time_ms": bundle["debug_execution_time_ms"],
            "original_markdown_path": markdown_path,
            "vectorization_payload": vector_payload,
        }


if __name__ == "__main__":
    splitter = MultilingualSplitter()

    mixed_text_lines = [
        "System status: OK.",
        "Das System verarbeitet eingehende Anfragen.",
        "User clicked submit and the form validated.",
        "Bitte pruefen Sie die Eingabewerte.",
        "The pipeline starts with preprocessing.",
        "Die Verarbeitung wurde erfolgreich abgeschlossen.",
        "Cache hit for segment index 12.",
        "Die Verbindung wurde getrennt.",
        "A fallback route was activated.",
        "Der Benutzer erhielt eine Warnung.",
        "Metrics are recorded every minute.",
        "Fehler beim Laden der Konfiguration.",
        "Retrying the request with backoff.",
        "Die Anfrage ist ungueltig.",
        "The system writes a new schema.",
        "Uebersetzung gestartet.",
        "Payload ready for vectorization.",
        "Die Sitzung wurde beendet.",
        "Monitoring reports no anomalies.",
        "Bitte erneut versuchen.",
    ]
    mixed_text = "\n".join(mixed_text_lines)
    print("for language detection without translation (translate=False):")
    text_result = splitter.process_text(mixed_text, document_id="CHECKPOINT_TEXT",translate=False)
    print("Translated text preview:")
    print(text_result)
    print("=" * 80)
    print("for language detection with translation (translate=True):")
    text_result = splitter.process_text(mixed_text, document_id="CHECKPOINT_TEXT")
    print("Translated text preview:")
    print(text_result)
    print("=" * 80)
    print("for building vectorization payload:")
    payload = splitter.build_vectorization_payload(
        text=mixed_text,
        document_id="DOC001",
        translate=True
    )
    print("Vectorization payload keys:")
    print(list(payload.keys()))