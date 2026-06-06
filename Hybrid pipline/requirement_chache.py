import database_connection
import translation_check
from langdetect import detect
from typing import List


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def process_segments(segments: List[str]) -> List[str]:
    processed_list = []

    for segment in segments:
        lang = detect_language(segment)

        # German → check DB cache
        if lang == "de":
            cached_result = database_connection.get_cached_translation(segment)
            processed_list.append(cached_result)

        # English or others → direct pass-through
        else:
            processed_list.append(segment)

    return processed_list


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
    "Der Benutzer muss sich anmelden."
]
    result = process_segments(segments)
    translated_result = translation_check.translate_documents_to_english(result)
    database_connection.insert_translation_cache(translated_result)
    print(translated_result)
    print("Completed successfully.")
