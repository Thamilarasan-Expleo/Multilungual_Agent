import math
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from openpyxl import Workbook

from databricks_api import OpenAI_Databricks_Embedding


def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over", "under",
    "shall", "must", "should", "need", "needs", "using", "use", "used", "able",
    "die", "der", "das", "und", "mit", "von", "auf", "eine", "einer", "eines",
    "ist", "sind", "werden", "wird", "kann", "können", "nicht", "nur",
    "customer", "kunden", "service", "system", "plattform", "platform",
    "user", "users", "benutzer", "benutzerin", "benutzers",
}


def extract_keywords(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9_-]{3,}", text)
    keywords = []
    for token in tokens:
        lower = token.lower()
        if lower in STOPWORDS:
            continue
        keywords.append(lower)
    return list(dict.fromkeys(keywords))


def keyword_recall_at_3(keywords: List[str], chunks: List[str]) -> float:
    if not keywords:
        return 0.0
    combined = " ".join(chunks).lower()
    hits = sum(1 for kw in keywords if kw in combined)
    return hits / len(keywords)


def embed_texts(embedding_model, texts: List[str]) -> np.ndarray:
    embeddings = embedding_model.get_text_embedding_batch(texts)
    return np.array(embeddings, dtype=np.float32)


def write_excel_report(
    output_path: str,
    rows: List[Tuple[str, List[str], List[float], float, float]],
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "retrieval_report"

    sheet.append(
        [
            "query",
            "top_1_chunk",
            "top_1_score",
            "top_2_chunk",
            "top_2_score",
            "top_3_chunk",
            "top_3_score",
            "avg_top3_score",
            "keyword_recall_at_3",
        ]
    )

    for query, chunks, scores, avg_score, recall in rows:
        sheet.append(
            [
                query,
                chunks[0],
                scores[0],
                chunks[1],
                scores[1],
                chunks[2],
                scores[2],
                avg_score,
                recall,
            ]
        )

    workbook.save(output_path)


def main() -> None:
    corpus_path = "Hybrid pipline/mixed_de_en_segment_splitter_fixture_corpus.txt"
    chunk_size = 1000
    overlap = 200
    output_path = "Hybrid pipline/retrieval_report.xlsx"

    queries = [
        "Customer Service Representative needs to locate a customer using Kundennummer, Nachname and E-Mail-Adresse with fuzzy search enabled.",
        "Compliance Officer reviewing Einwilligungsstatus, Herkunftssystem and historical consent records for audit verification.",
        "Search results for Kundennummer C12345 are taking longer than three seconds and Trefferliste pagination is not working correctly.",
        "Need all mandatory fields including Kundennummer, Vorname, Nachname and E-Mail-Adresse for customer onboarding process.",
        "Audit report export to PDF and Excel missing revisionssichere Änderungen from consent history.",
        "Customer profile creation request submitted through POST /api/kunde/anlegen returned HTTP 400 validation error.",
        "Dashboard should display active customers, offene Vorgänge and Compliance-Kennzahlen with automatic refresh.",
        "Review of Authentifizierung requirements using OAuth 2.0, OpenID Connect and Multi-Faktor-Authentifizierung.",
        "Duplicate Kundennummer detected in customer database despite uniqueness constraint requirements.",
        "Benutzer cannot search customers using Telefonnummer and Nachname simultaneously in the Kunden tab.",
        "Verification of logging requirements for Benutzeranmeldung, Kennwort warnings and Datenbankverbindung failures.",
        "Contract Service and Audit-Service integration using Apache Kafka event-driven communication.",
        "Data migration validation completed but Datenqualität monitoring remains active due to legacy system synchronization.",
        "Compliance review of personenbezogene Daten processing, Datenschutzrichtlinien and regulatory audit retention.",
        "Language switching between Deutsch and English not updating field labels such as Kundenname, Vertragsbeginn and Vertragsende.",
        "Customer onboarding workflow requiring customer master data, consent capture and contract initialization across multiple services.",
        "Support ticket regarding inaccessible Kundendetailseite while customer records remain available in backend systems.",
        "Historical Einwilligung records must remain immutable while audit reporting continues to provide full traceability.",
        "CustomerRelationshipManagementDatabase synchronization event processed successfully but reporting service data appears outdated.",
        "Review of role-based access permissions for Kundendienstmitarbeiter, Compliance-Beauftragter and Systemadministrator.",
    ]

    text = load_text(corpus_path)
    chunks = chunk_text(text, chunk_size, overlap)

    embedding_wrapper = OpenAI_Databricks_Embedding()
    embedding_model = embedding_wrapper.as_llama_embedding()

    chunk_embeddings = embed_texts(embedding_model, chunks)
    query_embeddings = embed_texts(embedding_model, queries)

    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    chunk_norm = chunk_embeddings / np.clip(norms, 1e-12, None)
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_norm = query_embeddings / np.clip(query_norms, 1e-12, None)

    similarity = query_norm @ chunk_norm.T

    rows = []
    avg_scores = []
    keyword_recalls = []

    for i, query in enumerate(queries):
        scores = similarity[i]
        ranked = np.argsort(scores)[::-1]
        top_indices = ranked[:3]
        top_chunks = [chunks[idx] for idx in top_indices]
        top_scores = [float(scores[idx]) for idx in top_indices]
        avg_top3 = float(np.mean(top_scores))

        keywords = extract_keywords(query)
        recall = keyword_recall_at_3(keywords, top_chunks)

        rows.append((query, top_chunks, top_scores, avg_top3, recall))
        avg_scores.append(avg_top3)
        keyword_recalls.append(recall)

    write_excel_report(output_path, rows)

    print("Retrieval Evaluation Report")
    print("Model: Databricks text-embedding-3-large")
    print(f"Chunks: {len(chunks)} | Chunk size: {chunk_size} | Overlap: {overlap}")
    print(f"Queries: {len(queries)}")
    print(f"Output: {output_path}")
    print("-")
    print(f"Average Top-3 Similarity: {float(np.mean(avg_scores)):.4f}")
    print(f"Average Keyword Recall@3: {float(np.mean(keyword_recalls)):.4f}")


if __name__ == "__main__":
    main()
