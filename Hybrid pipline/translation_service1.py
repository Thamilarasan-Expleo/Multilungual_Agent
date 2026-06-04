from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import torch
import time
 
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
            translated_group = translate_chunks_in_batches(
                grouped_texts,
                lang
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
 
    # german_text  = ['# Projektname: Globale Kundenmanagement-Plattform (GKMP)\n\n## Projektübersicht\n\nDie Globale Kundenmanagement-Plattform (GKMP) stellt eine zentrale Lösung für die Verwaltung von Kundeninformationen über verschiedene Regionen, Geschäftsbereiche und regulatorische Zuständigkeiten hinweg bereit.', '\n\nDie Lösung unterstützt deutschsprachige und englischsprachige Benutzer und ermöglicht eine nahtlose Integration mit bestehenden Unternehmensanwendungen.', '\n\nDas System muss hochverfügbar sein und eine Verarbeitung großer Datenmengen mit minimaler Latenz gewährleisten.', '\n\n---\n\n# Geschäftlicher Hintergrund\n\nDas Unternehmen betreibt derzeit mehrere voneinander unabhängige Anwendungen zur Speicherung von Kundendaten.', 'Dies führt zu Dateninkonsistenzen, redundanten Datensätzen, Synchronisierungsproblemen und erhöhtem Verwaltungsaufwand.', '\n\nMehrere Fachabteilungen haben Bedenken hinsichtlich Datenqualität, Berichtsverzögerungen und regulatorischer Compliance geäußert.', '\n\n---\n\n# Geschäftsziele\n\n## Primäre Ziele\n\n* Zentralisierung sämtlicher Kundendaten\n* Verbesserung der Datenqualität\n* Reduzierung redundanter Datensätze\n* Einhaltung regulatorischer Anforderungen\n* Unterstützung internationaler Geschäftsprozesse\n* Verbesserung der Such- und Analysefunktionen\n\n## Secondary Objectives\n\n* Faster customer onboarding\n* Improved reporting capabilities\n* Enhanced audit traceability\n* Better integration with third-party systems\n* Reduced operational overhead\n\n---\n\n# Benutzerrollen\n\n## Vertriebssachbearbeiter\n\nDer Vertriebssachbearbeiter verwaltet Kundenprofile, erstellt Verträge und pflegt Stammdaten.', '\n\n## Compliance-Beauftragter\n\nDer Compliance-Beauftragte überwacht regulatorische Anforderungen und prüft Einwilligungsnachweise.', '\n\n---\n\n# User Story US-001 – Kundensuche\n\n## Beschreibung\n\nAls Kundendienstmitarbeiter möchte ich Kunden anhand verschiedener Suchkriterien finden können, damit ich Kundeninformationen schnell abrufen kann.', '\n\n## Suchkriterien\n\nDer Benutzer soll Kunden anhand folgender Felder suchen können:\n\n* Kundennummer\n* Vorname\n* Nachname\n* E-Mail-Adresse\n* Telefonnummer\n* Geburtsdatum\n* Vertragsnummer\n* Kundenstatus\n\n## Akzeptanzkriterien\n\n* Das Feld "Kundennummer" muss durchsuchbar sein.', '\n* Das Feld "Nachname" muss unterstützt werden.', '\n* Das Feld "Vorname" muss unterstützt werden.', '\n* Die Trefferliste muss paginiert dargestellt werden.', '\n* Die Suche muss Groß- und Kleinschreibung ignorieren.', '\n* Suchergebnisse müssen nach Relevanz sortiert werden können.', '\n\n---\n\n# User Story US-002 – Einwilligungsverwaltung\n\n## Beschreibung\n\nAls Compliance-Beauftragter möchte ich sämtliche Kundeneinwilligungen einsehen können, damit regulatorische Anforderungen überprüft werden können.', '\n\n## Funktionale Anforderungen\n\nDer Compliance-Beauftragte muss folgende Informationen anzeigen können:\n\n* Einwilligungsstatus\n* Zeitstempel\n* Herkunftssystem\n* Änderungsverlauf\n* Verantwortlicher Benutzer\n\nThe system shall preserve historical records for auditing purposes.', '\n\n## Akzeptanzkriterien\n\n* Einwilligungshistorien dürfen nicht bearbeitet werden.', '\n* Änderungen müssen revisionssicher gespeichert werden.', '\n* Auditberichte müssen als PDF und Excel exportierbar sein.', '\n* Das System muss sämtliche Zugriffe protokollieren.', '\n* Jede Änderung muss einem Benutzer zugeordnet werden können.', '\n\n---\n\n# Technische Architektur\n\n## Architekturübersicht\n\nThe application follows a microservice architecture consisting of independently deployable services.', '\n\nJeder Service besitzt eine klar definierte Verantwortlichkeit und kommuniziert über standardisierte REST-Schnittstellen.', '\n\n## API-Gateway\n\nDas API-Gateway fungiert als zentraler Einstiegspunkt für sämtliche Client-Anfragen.', '\n\nFunktionen:\n\n* Authentifizierung\n* Autorisierung\n* Routing\n* Lastverteilung\n* API-Protokollierung\n\n## Authentifizierung\n\nDie Authentifizierung erfolgt über:\n\n* OAuth 2.0\n* OpenID Connect\n* Multi-Faktor-Authentifizierung\n\nThe system shall support integration with enterprise identity providers.', '\n\n---\n\n# Kerndienste\n\n## Kundenservice\n\nVerwaltung von Kundenstammdaten und Kundenbeziehungen.', '\n\n## Vertragsservice\n\nVerwaltung sämtlicher Vertragsinformationen.', '\n\n## Benachrichtigungsservice\n\nVersand von E-Mails, SMS-Nachrichten und Systembenachrichtigungen.', '\n\n## Audit-Service\n\nSpeicherung aller revisionsrelevanten Ereignisse.', '\n\n## Berichtsservice\n\nGenerierung operativer und regulatorischer Berichte.', '\n\n## Compliance-Service\n\nDer Compliance-Service überwacht regulatorische Anforderungen und Datenschutzrichtlinien.', '\n\n---\n\n# Ereignisgesteuerte Kommunikation\n\nDie Plattform verwendet Apache Kafka für die asynchrone Kommunikation zwischen Services.', '\n\nExamples of business events:\n\n* KundeErstellt\n* KundeAktualisiert\n* VertragAngelegt\n* EinwilligungErteilt\n* EinwilligungWiderrufen\n\n---\n\n# Datenbankanforderungen\n\n## Kundentabelle\n\nDie Kundendatenbank muss folgende Felder unterstützen:\n\n| Feldname|\n| --------------------------- |\n| Kundennummer                |\n| Vorname                     |\n| Nachname                    |\n| Geburtsdatum                |\n| E-Mail-Adresse              |\n| Telefonnummer               |\n| Kundenstatus                |\n| Einwilligungsstatus         |\n| Registrierungsdatum         |\n| LetztesAktualisierungsdatum |\n\n## Pflichtfelder\n\nFolgende Felder müssen verpflichtend sein:\n\n* Kundennummer\n* Vorname\n* Nachname\n* E-Mail-Adresse\n\n## Datenbankregeln\n\n* Kundennummern müssen eindeutig sein.', '\n* E-Mail-Adressen dürfen nicht doppelt vergeben werden.', '\n* Fremdschlüsselbeziehungen müssen konsistent gehalten werden.', '\n* Historische Daten dürfen nicht überschrieben werden.', '\n\n---\n\n# API-Anforderungen\n\n## Kunde erstellen\n\n### Endpunkt\n\nPOST /api/kunde/anlegen\n\n### Anfragefelder\n\n* Kundennummer\n* Vorname\n* Nachname\n* E-Mail-Adresse\n* Telefonnummer\n* Geburtsdatum\n\n### Erfolgreiche Antwort\n\nHTTP Status 201\n\n### Fehlerbehandlung\n\n* HTTP 400 – Ungültige Anfrage\n* HTTP 401 – Nicht autorisiert\n* HTTP 403 – Zugriff verweigert\n* HTTP 404 – Ressource nicht gefunden\n* HTTP 409 – Konflikt\n* HTTP 500 – Interner Serverfehler\n\nValidation errors shall contain detailed descriptions explaining the cause of the failure.', '\n\n## Registerkarten\n\n* Dashboard\n* Kunden\n* Verträge\n* Berichte\n* Einstellungen\n* Compliance\n* Administration\n\n## Dashboard\n\nDas Dashboard muss folgende Informationen anzeigen:\n\n* Anzahl aktiver Kunden\n* Anzahl neuer Kunden\n* Anzahl offener Vorgänge\n* Anzahl aktiver Verträge\n* Compliance-Kennzahlen\n* Systemstatus\n\nThe dashboard shall refresh data automatically every five minutes.', '\n\n---\n\n# Feldbezeichnungen\n\n## Kundendaten\n\n* Kundenname\n* Kundennummer\n* Kundenstatus\n* Vertragsstatus\n* Genehmigungsstatus\n* Vertragsbeginn\n* Vertragsende\n* E-Mail-Adresse\n* Telefonnummer\n* Geburtsdatum\n\n## Compliance-Daten\n\n* Einwilligungsstatus\n* Einwilligungsdatum\n* Herkunftssystem\n* Prüfstatus\n* Datenschutzklassifizierung\n\n---\n\n# Fehlerbehandlung\n\nDas System muss unerwartete Fehler kontrolliert behandeln.', '\n\n## Fehlermeldungen\n\n* Fehler 401: Nicht autorisiert.', '\n* Fehler 403: Zugriff verweigert.', '\n* Fehler 404: Datensatz nicht gefunden.', '\n* Fehler 409: Datensatzkonflikt.', '\n* Fehler 422: Validierungsfehler.', '\n* Fehler 500: Interner Serverfehler.', '\n\n---\n\n# Protokollierungsanforderungen\n\nAlle kritischen Geschäftsoperationen müssen protokolliert werden.', '\n\n## Beispielprotokolle\n\n2026-03-01 08:00:15 INFO Benutzeranmeldung erfolgreich.', '\n\n2026-03-01 08:00:18 INFO Benutzer erfolgreich authentifiziert.', '\n\n2026-03-01 08:02:30 WARN Kennwort läuft in 5 Tagen ab.', '\n\n2026-03-01 08:03:20 ERROR Datenbankverbindung fehlgeschlagen.', '\n\n2026-03-01 08:03:21 ERROR Kunde konnte nicht gespeichert werden.', '\n\n---\n\n# Datenschutz- und Compliance-Anforderungen\n\nGemäß den geltenden Datenschutzbestimmungen dürfen personenbezogene Daten ausschließlich für legitime Geschäftszwecke verarbeitet werden.', '\n\n## Sicherheitsanforderungen\n\n* Verschlüsselung ruhender Daten\n* Verschlüsselung übertragener Daten\n* Rollenbasierte Zugriffskontrolle\n* Multi-Faktor-Authentifizierung\n* Sicherheitsprotokollierung\n* Regelmäßige Sicherheitsüberprüfungen\n\nAlle Mitarbeiter sind verpflichtet, Datenschutzrichtlinien einzuhalten.', '\n\n---\n\n# Rechtliche Bestimmungen\n\nGemäß §14 Absatz 3 dieser Vereinbarung verpflichtet sich der Auftragnehmer, sämtliche vertraulichen Informationen vertraulich zu behandeln.', '\n\n---\n\n# Support-Ticket-Beispiel\n\n## Ticketnummer\n\nINC-1001\n\n## Benutzermeldung\n\n"Ich kann die Kundendetailseite nicht öffnen."', '\n\n## Support-Antwort\n\n"Das Problem wird derzeit untersucht."', '\n\n## Status\n\nIn Bearbeitung\n\n---\n\n# E-Mail-Beispiel\n\n## Betreff\n\nStatus der Kundendatenmigration\n\nSehr geehrtes Team,\n\nDie Migration wurde erfolgreich abgeschlossen.', 'Sämtliche Kundendatensätze wurden validiert und in die Zielumgebung importiert.', '\n\nMit freundlichen Grüßen\n\nProjektleitung\n\n---\n\n# Komplexes Szenario\n\nObwohl die Datenmigration erfolgreich abgeschlossen wurde und sämtliche Qualitätsprüfungen positiv verlaufen sind, muss das Projektteam weiterhin die Datenqualität überwachen, da einzelne Altsysteme noch nicht vollständig außer Betrieb genommen wurden, while several downstream applications continue synchronizing customer information across regional environments and compliance validation processes remain active.', '\n\n---\n\n# Gemischte Technische Aussage\n\nDie KundenprofilSynchronisierungsEngine verarbeitet eingehende Ereignisse aus mehreren Quellsystemen und aktualisiert die KundenbeziehungsManagementDatenbank, while the AuditComplianceMonitoringService records every transaction for regulatory reporting, operational monitoring, and long-term audit retention purposes.']
    german_text = [
"Projektname: Globale Kundenmanagement Plattform GKMP Die Globale Kundenmanagement Plattform GKMP stellt eine zentrale Lösung für die Verwaltung von Kundeninformationen über verschiedene Regionen, Geschäftsbereiche und regulatorische Zuständigkeiten hinweg bereit.",
"Die Lösung unterstützt deutschsprachige und englischsprachige Benutzer und ermöglicht eine nahtlose Integration mit bestehenden Unternehmensanwendungen.",
"Das System muss hochverfügbar sein und eine Verarbeitung großer Datenmengen mit minimaler Latenz gewährleisten.",
"Geschäftlicher Hintergrund Das Unternehmen betreibt derzeit mehrere voneinander unabhängige Anwendungen zur Speicherung von Kundendaten.",
"Dies führt zu Dateninkonsistenzen, redundanten Datensätzen, Synchronisierungsproblemen und erhöhtem Verwaltungsaufwand.",
"Mehrere Fachabteilungen haben Bedenken hinsichtlich Datenqualität, Berichtsverzögerungen und regulatorischer Compliance geäußert.",
"Geschäftsziele Primäre Ziele Zentralisierung sämtlicher Kundendaten Verbesserung der Datenqualität Reduzierung redundanter Datensätze Einhaltung regulatorischer Anforderungen Unterstützung internationaler Geschäftsprozesse Verbesserung der Such und Analysefunktionen.",
"Secondary Objectives Faster customer onboarding Improved reporting capabilities Enhanced audit traceability Better integration with third party systems Reduced operational overhead.",
"Benutzerrollen Vertriebssachbearbeiter Der Vertriebssachbearbeiter verwaltet Kundenprofile, erstellt Verträge und pflegt Stammdaten.",
"Compliance Beauftragter Der Compliance Beauftragte überwacht regulatorische Anforderungen und prüft Einwilligungsnachweise.",
"User Story US 001 Kundensuche Beschreibung Als Kundendienstmitarbeiter möchte ich Kunden anhand verschiedener Suchkriterien finden können, damit ich Kundeninformationen schnell abrufen kann.",
"Suchkriterien Der Benutzer soll Kunden anhand folgender Felder suchen können Kundennummer Vorname Nachname E Mail Adresse Telefonnummer Geburtsdatum Vertragsnummer Kundenstatus.",
"Akzeptanzkriterien Das Feld Kundennummer muss durchsuchbar sein.",
"Das Feld Nachname muss unterstützt werden.",
"Das Feld Vorname muss unterstützt werden.",
"Die Trefferliste muss paginiert dargestellt werden.",
"Die Suche muss Groß und Kleinschreibung ignorieren.",
"Suchergebnisse müssen nach Relevanz sortiert werden können.",
"User Story US 002 Einwilligungsverwaltung Beschreibung Als Compliance Beauftragter möchte ich sämtliche Kundeneinwilligungen einsehen können, damit regulatorische Anforderungen überprüft werden können.",
"Funktionale Anforderungen Der Compliance Beauftragte muss folgende Informationen anzeigen können Einwilligungsstatus Zeitstempel Herkunftssystem Änderungsverlauf Verantwortlicher Benutzer.",
"The system shall preserve historical records for auditing purposes.",
"Akzeptanzkriterien Einwilligungshistorien dürfen nicht bearbeitet werden.",
"Änderungen müssen revisionssicher gespeichert werden.",
"Auditberichte müssen als PDF und Excel exportierbar sein.",
"Das System muss sämtliche Zugriffe protokollieren.",
"Jede Änderung muss einem Benutzer zugeordnet werden können.",
"Technische Architektur Architekturübersicht The application follows a microservice architecture consisting of independently deployable services.",
"Jeder Service besitzt eine klar definierte Verantwortlichkeit und kommuniziert über standardisierte REST Schnittstellen.",
"API Gateway Das API Gateway fungiert als zentraler Einstiegspunkt für sämtliche Client Anfragen.",
"Funktionen Authentifizierung Autorisierung Routing Lastverteilung API Protokollierung.",
"Authentifizierung Die Authentifizierung erfolgt über OAuth 2.0 OpenID Connect Multi Faktor Authentifizierung.",
"The system shall support integration with enterprise identity providers.",
"Kerndienste Kundenservice Verwaltung von Kundenstammdaten und Kundenbeziehungen.",
"Vertragsservice Verwaltung sämtlicher Vertragsinformationen.",
"Benachrichtigungsservice Versand von E Mails SMS Nachrichten und Systembenachrichtigungen.",
"Audit Service Speicherung aller revisionsrelevanten Ereignisse.",
"Berichtsservice Generierung operativer und regulatorischer Berichte.",
"Compliance Service Der Compliance Service überwacht regulatorische Anforderungen und Datenschutzrichtlinien.",
"Ereignisgesteuerte Kommunikation Die Plattform verwendet Apache Kafka für die asynchrone Kommunikation zwischen Services.",
"Examples of business events KundeErstellt KundeAktualisiert VertragAngelegt EinwilligungErteilt EinwilligungWiderrufen.",
"Datenbankanforderungen Kundentabelle Die Kundendatenbank muss folgende Felder unterstützen Kundennummer Vorname Nachname Geburtsdatum E Mail Adresse Telefonnummer Kundenstatus Einwilligungsstatus Registrierungsdatum LetztesAktualisierungsdatum.",
"Pflichtfelder Folgende Felder müssen verpflichtend sein Kundennummer Vorname Nachname E Mail Adresse.",
"Datenbankregeln Kundennummern müssen eindeutig sein.",
"E Mail Adressen dürfen nicht doppelt vergeben werden.",
"Fremdschlüsselbeziehungen müssen konsistent gehalten werden.",
"Historische Daten dürfen nicht überschrieben werden.",
"API Anforderungen Kunde erstellen Endpunkt POST api kunde anlegen.",
"Anfragefelder Kundennummer Vorname Nachname E Mail Adresse Telefonnummer Geburtsdatum.",
"Erfolgreiche Antwort HTTP Status 201.",
"Fehlerbehandlung HTTP 400 Ungültige Anfrage HTTP 401 Nicht autorisiert HTTP 403 Zugriff verweigert HTTP 404 Ressource nicht gefunden HTTP 409 Konflikt HTTP 500 Interner Serverfehler.",
"Validation errors shall contain detailed descriptions explaining the cause of the failure.",
"Registerkarten Dashboard Kunden Verträge Berichte Einstellungen Compliance Administration.",
"Dashboard Das Dashboard muss folgende Informationen anzeigen Anzahl aktiver Kunden Anzahl neuer Kunden Anzahl offener Vorgänge Anzahl aktiver Verträge Compliance Kennzahlen Systemstatus.",
"The dashboard shall refresh data automatically every five minutes.",
"Feldbezeichnungen Kundendaten Kundenname Kundennummer Kundenstatus Vertragsstatus Genehmigungsstatus Vertragsbeginn Vertragsende E Mail Adresse Telefonnummer Geburtsdatum.",
"Compliance Daten Einwilligungsstatus Einwilligungsdatum Herkunftssystem Prüfstatus Datenschutzklassifizierung.",
"Fehlerbehandlung Das System muss unerwartete Fehler kontrolliert behandeln.",
"Fehlermeldungen Fehler 401 Nicht autorisiert Fehler 403 Zugriff verweigert Fehler 404 Datensatz nicht gefunden Fehler 409 Datensatzkonflikt Fehler 422 Validierungsfehler Fehler 500 Interner Serverfehler.",
"Protokollierungsanforderungen Alle kritischen Geschäftsoperationen müssen protokolliert werden.",
"Beispielprotokolle 2026 03 01 08 00 15 INFO Benutzeranmeldung erfolgreich 2026 03 01 08 00 18 INFO Benutzer erfolgreich authentifiziert 2026 03 01 08 02 30 WARN Kennwort läuft in 5 Tagen ab 2026 03 01 08 03 20 ERROR Datenbankverbindung fehlgeschlagen 2026 03 01 08 03 21 ERROR Kunde konnte nicht gespeichert werden.",
"Datenschutz und Compliance Anforderungen Gemäß den geltenden Datenschutzbestimmungen dürfen personenbezogene Daten ausschließlich für legitime Geschäftszwecke verarbeitet werden.",
"Sicherheitsanforderungen Verschlüsselung ruhender Daten Verschlüsselung übertragener Daten Rollenbasierte Zugriffskontrolle Multi Faktor Authentifizierung Sicherheitsprotokollierung Regelmäßige Sicherheitsüberprüfungen.",
"Alle Mitarbeiter sind verpflichtet, Datenschutzrichtlinien einzuhalten.",
"Rechtliche Bestimmungen Gemäß 14 Absatz 3 dieser Vereinbarung verpflichtet sich der Auftragnehmer, sämtliche vertraulichen Informationen vertraulich zu behandeln.",
"Support Ticket Beispiel Ticketnummer INC 1001.",
"Benutzermeldung Ich kann die Kundendetailseite nicht öffnen.",
"Support Antwort Das Problem wird derzeit untersucht.",
"Status In Bearbeitung.",
"E Mail Beispiel Betreff Status der Kundendatenmigration.",
"Sehr geehrtes Team Die Migration wurde erfolgreich abgeschlossen Sämtliche Kundendatensätze wurden validiert und in die Zielumgebung importiert Mit freundlichen Grüßen Projektleitung.",
"Komplexes Szenario Obwohl die Datenmigration erfolgreich abgeschlossen wurde und sämtliche Qualitätsprüfungen positiv verlaufen sind, muss das Projektteam weiterhin die Datenqualität überwachen, da einzelne Altsysteme noch nicht vollständig außer Betrieb genommen wurden, while several downstream applications continue synchronizing customer information across regional environments and compliance validation processes remain active.",
"Gemischte Technische Aussage Die KundenprofilSynchronisierungsEngine verarbeitet eingehende Ereignisse aus mehreren Quellsystemen und aktualisiert die KundenbeziehungsManagementDatenbank, while the AuditComplianceMonitoringService records every transaction for regulatory reporting, operational monitoring, and long term audit retention purposes."
]
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
        segment["text"]
        for segment in segments
        if segment.get("language") == "de" ]
    result = translate_documents_to_english(german_text)
 
    print("\nTranslated Output:\n")
    print("length of Translated Output:", len(result))
    print(result)
 
 