import database_connection
import translation_function
from langdetect import detect
from typing import List


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def process_segments(segments):

    final_results = [None] * len(segments)

    texts_to_translate = []
    translate_indexes = []

    for idx, segment in enumerate(segments):

        lang = detect_language(segment)

        # English → no translation required
        if lang == "en":
            final_results[idx] = segment
            continue

        cached_translation = (
            database_connection.get_cached_translation(segment)
        )

        if cached_translation:

            # Cache hit
            final_results[idx] = cached_translation

        else:

            # Cache miss
            texts_to_translate.append(segment)
            translate_indexes.append(idx)

    return (
        final_results,
        texts_to_translate,
        translate_indexes
    )

# def process_segments(segments: List[str]) -> List[str]:
#     processed_list = []

#     for segment in segments:
#         lang = detect_language(segment)

#         # German → check DB cache
#         if lang == "de":
#             cached_result = database_connection.get_cached_translation(segment)
#             processed_list.append(cached_result)

#         # English or others → direct pass-through
#         else:
#             processed_list.append(segment)

#     return processed_list


if __name__ == "__main__":
    segments = [
    "Projektübersicht: Die Global Customer Management Platform (GCMP) soll eine zentrale Lösung für die Verwaltung von Kundendaten über verschiedene Regionen und Geschäftsbereiche hinweg bereitstellen.",
    "The platform should provide a unified view of customer information across all business units.",
    "Das System muss real-time data synchronization unterstützen, damit Änderungen an Kundeninformationen sofort in allen verbundenen Anwendungen verfügbar sind.",
    "Benutzer mit entsprechenden Berechtigungen können Kundenprofile erstellen, aktualisieren, durchsuchen und archivieren.",
    "Customer records must be searchable using advanced filtering and sorting capabilities.",
    "Die Plattform soll eine sichere user authentication und role-based access control implementieren, um den Zugriff auf sensible Informationen zu schützen.",
    "Darüber hinaus muss das System eine vollständige Historie aller Änderungen im audit trail speichern, sodass jede Aktion nachvollziehbar bleibt.",
    "The system should generate notifications whenever critical customer information is modified.",
    "Das Dashboard soll interaktive analytics reports, KPI-Übersichten und individuelle Filtermöglichkeiten bereitstellen.",
    "Die Anwendung muss Daten aus bestehenden CRM systems, ERP-Lösungen und externen APIs integrieren.",
    "Integration services should support both batch processing and real-time communication.",
    "Für eine bessere Benutzererfahrung soll die Plattform responsive design unterstützen und auf Desktop-, Tablet- und Mobilgeräten nutzbar sein.",
    "Zusätzlich müssen alle Datenverarbeitungsprozesse den Anforderungen der GDPR compliance entsprechen.",
    "All sensitive customer data must be encrypted both at rest and in transit.",
    "Das System soll eine hohe Verfügbarkeit, automatische Backups, error monitoring sowie skalierbare Cloud-Infrastruktur bereitstellen, um zukünftiges Wachstum und steigende Benutzerzahlen effizient zu unterstützen.",
    "The application should maintain an uptime of at least 99.9 percent under normal operating conditions.",
    "Der Benutzer muss sich anmelden.",
    "Das System muss automatisch Benachrichtigungen senden, wenn kritische Fehler erkannt werden.",
    "Benutzer sollen Berichte im PDF- und Excel-Format exportieren können.",
    "Die Anwendung muss eine mehrsprachige Benutzeroberfläche für internationale Kunden bereitstellen."

]
    final_results, texts_to_translate, translate_indexes = process_segments(segments)
    # Translate only cache misses
    if texts_to_translate:
        print("Newly identified non-English segments requiring translation:")

        translated_records = (
            translation_function.translate_documents_to_english(
                texts_to_translate
            )
        )
        # Insert newly translated records into DB
        database_connection.insert_translation_cache(
            translated_records
        )
        # Put translations back in original positions
        for idx, record in zip(
            translate_indexes,
            translated_records
        ):
            final_results[idx] = record["translated_text"]

    print(final_results)
    print("Completed successfully.")
