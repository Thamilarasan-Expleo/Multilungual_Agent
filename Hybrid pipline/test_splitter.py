import math
import os
import logging
import openpyxl
from dotenv import load_dotenv
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from splitter import MultilingualSplitter

load_dotenv()

MOUNTED_FOLDER = os.getenv("MOUNTED_FOLDER", "translated_data")
os.makedirs(MOUNTED_FOLDER, exist_ok=True)

LOG_LEVEL = os.getenv("SPLITTER_TEST_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# CONFIGURABLE ACCURACY TUNING PARAMETER
# Set this between 2.0 and 3.0 to expand your acceptable boundary tolerance window
SIGMA_MULTIPLIER = float(os.getenv("SPLITTER_SIGMA_MULTIPLIER", "2.0"))  

TEST_DATA_FILENAME = os.getenv(
    "SPLITTER_TEST_DATA_FILENAME",
    "mixed_de_en_segment_splitter_fixture_corpus.txt",
)

def load_test_data_fixture():
    data_path = os.path.join(MOUNTED_FOLDER, TEST_DATA_FILENAME)
    if not os.path.exists(data_path):
        logger.error("Test data fixture not found at %s", data_path)
        return None
    with open(data_path, "r", encoding="utf-8") as handle:
        data_text = handle.read().strip()
    if not data_text:
        logger.error("Test data fixture is empty at %s", data_path)
        return None
    return {
        "id": "TEST_DATA",
        "category": "External Test Data",
        "text": data_text,
        "description": f"Loaded from {TEST_DATA_FILENAME}"
    }
# 16 High-Density User Stories (>500 characters each)
# test_fixtures = [
#     {"id": "US-01", "category": "Core Authentication UI", "text": "As a Platform End-User, I want a secure enterprise multi-tenant logging infrastructure. Requirements: Das System muss ein ansprechendes, reaktionsschnelles Frontend-Interface bereitstellen. Wenn der Benutzer auf den 'Submit Button' klickt, muss eine verschlüsselte Payload-Validierung gestartet werden. Das System prüft die Felder 'userId', 'tenantId' und 'sessionToken'. Falls die Authentifizierung fehlschlägt, wird eine benutzerdefinierte Fehlermeldung ausgegeben: »Der Authentifizierungsvorgang schlug aufgrund ungültiger Anmeldedaten fehl.« The interface must gracefully handle OAuth2 fallback paths automatically to prevent session termination blocks.", "description": "Mixed strings containing German guillemets and technical JSON keys."},
#     {"id": "US-02", "category": "Regulatory Compliance Engine", "text": "As an Enterprise Compliance Officer, I want the system to continuously validate incoming telemetry datasets against European data frameworks. Specification: Alle eingehenden Datensätze müssen strikt die gesetzliche Schadstoffentsorgungsrichtlinie erfüllen, bevor sie im cloud-basierten Data-Warehouse gespeichert werden. Die automatisierte Überprüfung analysiert spezifische Metriken wie 'carbon_footprint_index' und 'environmental_impact_score'. Falls Abweichungen im Audit-Log festgestellt werden, muss das System die Datenübertragung sofort einfrieren. Die interne DSGVO-Konformitätsprüfung löst daraufhin eine administrative Sicherheitswarnung aus, um potenzielle rechtliche Haftungsrisiken für die Organisation proaktiv zu minimieren.", "description": "Long German compound words mixed with English tracking parameters."},
#     {"id": "US-03", "category": "Data Engineering Pipeline", "text": "As a Senior Data Engineer, I want a high-performance vector ingestion pipeline for unstructured document parsing. Pipeline Logic: Die ETL-Pipeline muss Dokumente im PDF- und XML-Format automatisiert einlesen. Während der Vorverarbeitung (Pre-processing Phase) extrahiert das System den Text und führt eine Segmentierung durch. Ein integrierter BM25-Ranking-Algorithmus bewertet die Relevanz der extrahierten Textpassagen. Falls ein Segment gemischten Text enthält, muss das System das Flag 'lang' auf 'de' setzen, damit die nachgelagerte Übersetzungsschicht (Translation Layer) den Text fehlerfrei in ein englisches Einbettungsmodell überführen kann. This prevents embedding distortion and indexing drops.", "description": "Technical RAG architectural specifications and data pipelines."},
#     {"id": "US-04", "category": "API Gateway Routing", "text": "As a Third-Party API Developer, I want a rate-limited RESTful API Gateway endpoint to query transaction data. Route Definition: Dieser hocheffiziente Endpunkt führt ein tiefes semantisches Parsing für jede Anfrage durch. Es akzeptiert optionale Payload-Parameter wie 'max_length', 'temperature' oder 'top_p' Werte. Die Geschäftslogik verarbeitet die Anfragen sequenziell: Gemäß Absatz 3 der internen API-Sicherheitsrichtlinie muss jede eingehende 'request_id' verifiziert werden. Wenn die Validierung fehlschlägt, wirft die Anwendungsschicht sofort einen '500 Internal Server Error' (Fehler beim Verbindungsaufbau) aus. The gateway must guarantee sub-50ms roundtrip responses under high concurrency.", "description": "API payload processing rules with inline parenthetical hints."},
#     {"id": "US-05", "category": "Asynchronous Event Bus", "text": "As a DevOps Infrastructure Engineer, I want an asynchronous event-driven system architecture to handle real-time logging telemetry. System Blueprint: Der Cloud-Infrastruktur-Bus empfängt kontinuierlich Telemetriedaten von verteilten Microservices. Der aktuelle Systemstatus lautet standardmäßig 'System Online'. Falls kritische Fehler auftreten, initialisiert das System sofort die Komponente 'Core Routing Component'. Jede Nachricht enthält Statuswerte wie 'Authenticated' oder 'Pending'. Die automatische Skalierungsrichtlinie startet daraufhin zusätzliche Worker-Instanzen: Initialisiere Core. Authenticated. Starten. Der gesamte Vorgang wird im globalen Cluster-Dashboard protokolliert, um Engpässe zu vermeiden.", "description": "Alternating sequence of brief English telemetry and German execution directives."},
#     {"id": "US-06", "category": "Legal Auditing Protocols", "text": "As a Corporate Legal Auditor, I want the document processing framework to parse contracts containing cross-lingual field definitions. Auditing Criteria: Das System muss Verträge automatisch nach vordefinierten Klauseln durchsuchen. Zu den Systemvoraussetzungen gehören z.B. high-end GPUs, modern network switches und performante Datenbankcluster. Gemäß den vertraglichen Vereinbarungen müssen geschäftskritische Felder wie 'liability_cap' und 'indemnity_clause' präzise extrahiert werden. Wenn Unstimmigkeiten in der Klauselstruktur erkannt werden, stoppt das System den Workflow. Der Benutzer erhält die Meldung: »Die Vertragsprüfung wurde aufgrund fehlender Pflichtangaben abgebrochen.« Please review logs.", "description": "Validates inline abbreviations like 'z.B.' to prevent false fractures."},
#     {"id": "US-07", "category": "Git GitOps Deployment", "text": "As a Release Manager, I want an automated GitOps deployment pipeline using automated canary verification steps. Deployment Script: Der CI/CD-Workflow überwacht das Repository kontinuierlich auf neue Commits. Ein typischer Ablauf lautet: Merged branch 'feature/auth' changes into main deployment branch. Der Build schlug jedoch fehl due to missing configurations inside the application properties matrix. Die Pipeline bricht ab und sendet eine Benachrichtigung an das Entwicklungsteam: »Kritischer Fehler während der Container-Orchestrierung festgestellt.« Das System führt automatisch ein Rollback durch, um die Stabilität der Produktivumgebung zu gewährleisten.", "description": "Combines raw source repository logs and conversational German descriptions."},
#     {"id": "US-08", "category": "Financial Ledger Processing", "text": "As an Enterprise Accounting Application, I want a secure financial ledger mechanism to process international invoicing requests. Transaction Flow: The billing module processes corporate accounts globally. Nach der erfolgreichen Validierung wird die Abrechnung ausgeführt: Der Gesamtbetrag inklusive MwSt. wurde erfolgreich abgebucht via authorized payment gateways. Das System generiert daraufhin ein PDF-Invoicing-Dokument für den Endkunden. Alle steuerrelevanten Datenpunkte werden in der relationalen PostgreSQL-Datenbank verschlüsselt gespeichert, um die lückenlose Konformität mit internationalen Rechnungslegungsvorschriften zu sichern.", "description": "Validates text configurations containing abbreviated ledger components like 'MwSt.'"},
#     {"id": "US-09", "category": "Agentic AI Execution", "text": "As an AI Architect, I want an intelligent Agentic RAG framework capable of orchestrating tools dynamically based on context. Agent Workflow: The autonomous agent continuously evaluates user input to determine intent. Unser Team nutzt spezialisierte Agentic RAG Frameworks für komplexe medizinische Abfragen. Wenn eine Anfrage eingeht, entscheidet der Agent über den nächsten Schritt. Ein typisches Szenario wäre: Das macht Sinn, right? Falls zusätzliche Fachdaten benötigt werden, ruft der Agent eine externe Wissensdatenbank auf. Die Antwortsegmente werden aggregiert, syntaktisch überprüft und als strukturierte JSON-Payload an das User-Interface zurückgegeben.", "description": "System operational workflows with embedded colloquial queries."},
#     {"id": "US-10", "category": "Front-End Dashboard Layout", "text": "As a Front-End UX Developer, I want a localized metrics dashboard layout to visualize system load anomalies. UI Specifications: Das Dashboard muss Echtzeit-Diagramme für CPU- und Speicherauslastung anzeigen. Klicken Sie auf 'Submit Button', um das Formular direkt abzusenden und die Filterparameter zu übernehmen. This action will automatically refresh your dashboard view and clear cached telemetry profiles. Falls der Web-Socket-Stream unterbricht, versucht die Client-Anwendung eine automatische Wiederverbindung. Der Ladeindikator zeigt währenddessen den aktuellen Verbindungsstatus an, um eine optimale Benutzererfahrung zu gewährleisten.", "description": "Interactive front-end operations blueprints mixed with English keys."},
#     {"id": "US-11", "category": "Database Performance Tuning", "text": "As a Database Administrator, I want automated execution profiling for slow-running analytical queries. DBA Strategy: The optimization engine runs queries through an explain-plan analysis layer. Refactored module 401: Fehlerbehebung im Core routing component durchgeführt, um Index-Scans zu optimieren. The processing latency dropped down dramatically from 45ms to 2ms after applying the missing database constraints. Das System überwacht nun kontinuierlich die Abfrage-Performance. Falls die Latenz erneut den definierten Schwellenwert überschreitet, wird automatisch ein administratives Ticket im Jira-System erstellt.", "description": "System refactoring descriptions tracking execution timing profiles."},
#     {"id": "US-12", "category": "Medical Query Knowledge Base", "text": "As a Clinical Research Data Analyst, I want a medical chatbot system capable of evaluating unstructured clinical summaries. Chatbot Requirement: The clinical intelligence module parses health documents securely. Das System fungiert als intelligenter medizinischer Abfrage-Chatbot namens 'MI Bot', der auf dem LangGraph-Framework basiert. Es analysiert komplexe Patientenakten und extrahiert Diagnosen. Kriterien werden hiermit validiert: Alle extrahierten medizinischen Fachbegriffe müssen mit standardisierten Katalogen abgeglichen werden. Die Ergebnisse werden strukturiert aufbereitet und dem behandelnden Arzt im Dashboard zur finalen Verifizierung angezeigt.", "description": "Medical chatbot configuration language containing project titles."},
#     {"id": "US-13", "category": "Enterprise Security Firewall", "text": "As a Cybersecurity Engineer, I want an automated threat detection mechanism to monitor incoming network packets. Security Policy: The boundary firewall inspects traffic payload patterns in real time. Das System blockiert verdächtige IP-Adressen automatisch beim Erkennen von Anomalien. Ein Vorfall wird wie folgt protokolliert: »Kritischer Sicherheitsalarm: Ein potenzieller Brute-Force-Angriff wurde im Systemnetzwerk identifiziert.« The monitoring agent immediately dispatches encrypted alert payloads to the security operations center via secure webhooks to trigger immediate mitigation protocols.", "description": "Validates system logs containing nested alert metrics."},
#     {"id": "US-14", "category": "Asynchronous Task Scheduler", "text": "As a Backend Systems Developer, I want a distributed cron execution matrix to trigger clean-up tasks during low-load hours. Architecture Logic: The microservice worker coordinates data storage compression routines. Das Hintergrund-Task-Management-System läuft vollständig asynchron. Der Scheduler startet den Bereinigungsprozess jede Nacht um 02:00 Uhr: Lösche temporäre Tabellen. Optimiere Datenbankindizes. Bereinige abgelaufene Benutzersitzungen. If an exception occurs during execution, the worker writes the complete stack trace to the centralized monitoring platform and retries the task up to three times.", "description": "Validates a quick series of single-clause technical command definitions."},
#     {"id": "US-15", "category": "E-Commerce Cart Processing", "text": "As an Online Store Front-End Engineer, I want a real-time shopping cart validation layer to manage pricing variations. Functional Steps: The checkout pipeline recalculates price adjustments dynamically. Das System validiert den aktuellen Warenkorb des Kunden während des Bezahlvorgangs. Es prüft Produktverfügbarkeiten: Alle ausgewählten Artikel müssen im Lagerbestand vorhanden sein, um Bestellverzögerungen zu vermeiden. If a product is out of stock, the interface displays an alert: 'Item unavailable'. Der Kunde wird gebeten, die Artikelstückzahl anzupassen oder ein alternatives Produkt auszuwählen.", "description": "Evaluates transitional store rules changing languages mid-block."},
#     {"id": "US-16", "category": "Cloud Architecture Core", "text": "As a Lead Solutions Architect, I want clear infrastructure milestone definitions to guide the multi-region cluster migration strategy. Milestones Checklist: Let's review the final cloud product architecture delivery milestones before starting the production deployment. Das Projekt wurde pünktlich abgeschlossen und alle Abnahmekriterien wurden erfolgreich erfüllt! Are there any architectural questions or configuration bottlenecks remaining? Falls keine Einwände erhoben werden, wird das Deployment-Skript für die europäische Kernregion unverzüglich ausgeführt.", "description": "Tests text entries ending in complex mixed characters (!, ?, .)."}
# ]

test_data_fixture = load_test_data_fixture()
test_fixtures = [item for item in [test_data_fixture] if item]

if not test_fixtures:
    logger.error("No test fixtures available. Check MOUNTED_FOLDER and fixture file name.")
    raise SystemExit(1)
# if test_data_fixture:
#     test_fixtures.append(test_data_fixture)

logger.info("Initializing Multilingual Range-Based Evaluation Framework...")
splitter = MultilingualSplitter()

# -------------------------------------------------------------------------
# PASS 1: Statistical Profiling Phase
# Run a baseline check over the corpus to compute the character distribution metrics
# -------------------------------------------------------------------------
char_per_segment_samples = []
raw_base_data = []

for item in test_fixtures:
    schema_path = os.path.join(MOUNTED_FOLDER, f"schema_{item['id']}.json")
    res = splitter.process_document(item["text"], document_id=item["id"], schema_output_path=schema_path, translate=True)
    actual_count = len(res["segments"])
    char_count = len(item["text"])
    
    # Store chars-per-segment ratio for this fixture
    ratio = char_count / actual_count if actual_count > 0 else char_count
    char_per_segment_samples.append(ratio)
    raw_base_data.append((actual_count, char_count, res["segments"], res["debug_execution_time_ms"]))

# Compute Population Mean (μ)
mean_ratio = sum(char_per_segment_samples) / len(char_per_segment_samples)

# Compute Population Standard Deviation (σ)
variance_sum = sum((x - mean_ratio) ** 2 for x in char_per_segment_samples)
std_deviation = math.sqrt(variance_sum / len(char_per_segment_samples))

logger.info("--- Statistical Baseline Captured ---")
logger.info("Mean Character Density per Segment (mu): %.2f", mean_ratio)
logger.info("Standard Deviation Vector Variation (sigma): %.2f", std_deviation)
logger.info("Target Sigma Window Range Configured  : +- %s sigma", SIGMA_MULTIPLIER)

# -------------------------------------------------------------------------
# PASS 2: Dynamic Sigma Range Boundaries Mapping & Evaluation
# -------------------------------------------------------------------------
processed_results = []

for idx, item in enumerate(test_fixtures):
    actual_count, char_count, segments_data, latency_ms = raw_base_data[idx]
    
    # Calculate mathematically expected center point segments count for this specific document length
    calculated_center = char_count / mean_ratio
    
    # Propagate the standard deviation variance across this item's specific structural size profile
    scaled_std_dev = (char_count * std_deviation) / (mean_ratio ** 2)
    boundary_offset = SIGMA_MULTIPLIER * scaled_std_dev
    
    # Dynamically determine the Min/Max bounds, enforcing a floor threshold of 1 segment
    min_expected = max(1, math.floor(calculated_center - boundary_offset))
    max_expected = max(min_expected + 1, math.ceil(calculated_center + boundary_offset))
    
    # Run Evaluation
    is_accurate = (min_expected <= actual_count <= max_expected)
    status = "PASSED" if is_accurate else "FAILED"
    
    logger.info(
        "[%s] Chars: %s | Allowed Range: %s-%s | Actual Segments: %s | %s",
        item["id"],
        char_count,
        min_expected,
        max_expected,
        actual_count,
        status,
    )
    
    processed_results.append({
        "id": item["id"], "category": item["category"], "text": item["text"],
        "min_expected": min_expected, "max_expected": max_expected,
        "actual_segments": actual_count, "status": status,
        "latency_ms": latency_ms, "segments_data": segments_data,
        "description": item["description"], "char_count": char_count
    })

# Run the 5000+ Word Performance Stress Matrix
stress_text = os.getenv(
    "SPLITTER_STRESS_TEXT",
    "The quick brown fox jumps over the lazy dog. »Dies ist ein deutscher Satz im Dokument.« ",
)
stress_multiplier = int(os.getenv("SPLITTER_STRESS_MULTIPLIER", "300"))
large_text = stress_text * stress_multiplier
stress_schema_path = os.path.join(MOUNTED_FOLDER, "schema_STRESS_5000.json")
stress_res = splitter.process_document(large_text, document_id="STRESS_5000", schema_output_path=stress_schema_path, translate=False)
logger.info(
    "stress test for 5000+ word document completed with latency: %.3f ms",
    stress_res["debug_execution_time_ms"],
)
# -------------------------------------------------------------------------
# PASS 3: Generate the Spreadsheet
# -------------------------------------------------------------------------
wb = openpyxl.Workbook()
ws_dash = wb.active
ws_dash.title = "Sprint Dashboard"
ws_dash.views.sheetView[0].showGridLines = True

HEADER_FILL = PatternFill("solid", fgColor="1F497D")
PASS_FILL, FAIL_FILL = PatternFill("solid", fgColor="E2EFDA"), PatternFill("solid", fgColor="FCE4D6")
ACCENT_FILL, ZEBRA_FILL, WHITE_FILL = PatternFill("solid", fgColor="DCE6F1"), PatternFill("solid", fgColor="F2F5F8"), PatternFill("solid", fgColor="FFFFFF")

FONT_TITLE = Font(name="Segoe UI", size=16, bold=True, color="1F497D")
FONT_SECTION = Font(name="Segoe UI", size=12, bold=True)
FONT_HEADER = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
FONT_BODY, FONT_BODY_BOLD = Font(name="Segoe UI", size=10), Font(name="Segoe UI", size=10, bold=True)
FONT_PASS, FONT_FAIL = Font(name="Segoe UI", size=10, bold=True, color="375623"), Font(name="Segoe UI", size=10, bold=True, color="C65911")
THIN_BORDER = Border(left=Side(style='thin', color='D9D9D9'), right=Side(style='thin', color='D9D9D9'), top=Side(style='thin', color='D9D9D9'), bottom=Side(style='thin', color='D9D9D9'))

ws_dash["B2"] = "Dynamic Sigma Segmentation Dashboard"
ws_dash["B2"].font = FONT_TITLE
ws_dash["B4"] = "Sprint Validation Summary Metrics"
ws_dash["B4"].font = FONT_SECTION

metrics = [
    ("Total Agile User Stories Evaluated", len(processed_results)),
    ("Passed Range Verification Tests", f"=COUNTIF('User Story Dynamic Logs'!H5:H20, \"PASSED\")"),
    ("Failed Range Verification Tests", f"=COUNTIF('User Story Dynamic Logs'!H5:H20, \"FAILED\")"),
    ("Overall Model Split Accuracy Score", f"=C6/C5"),
    ("5,000+ Word Performance Latency Profile", f"{stress_res['debug_execution_time_ms']:.3f} ms"),
    ("Strict SLA Metric Performance (<200ms)", "PASSED" if stress_res["debug_execution_time_ms"] < 200 else "FAILED")
]

for idx, (lbl, val) in enumerate(metrics, start=5):
    ws_dash.cell(row=idx, column=2, value=lbl).font = FONT_BODY_BOLD
    ws_dash.cell(row=idx, column=2).fill = ACCENT_FILL
    ws_dash.cell(row=idx, column=2).border = THIN_BORDER
    c = ws_dash.cell(row=idx, column=3, value=val)
    c.font = FONT_BODY
    c.border = THIN_BORDER
    if idx == 8: c.number_format = "0.0%"
    if "PASSED" in str(val) or "Passed" in lbl or "%" in lbl: c.font, c.fill = FONT_PASS, PASS_FILL
    if "FAILED" in str(val) or "Failed" in lbl: c.font, c.fill = FONT_FAIL, FAIL_FILL

# Sheet 2: Dynamic Execution Logs
ws_fixtures = wb.create_sheet(title="User Story Dynamic Logs")
ws_fixtures.views.sheetView[0].showGridLines = True
headers = ["Story ID", "Agile Feature Domain", "Requirement User Story Text (DE/EN Mixed)", "Char Count", "Sigma Min Bound", "Sigma Max Bound", "Actual Segments", "Status", "Latency (ms)", "Linguistic Split Category Objective"]

for c_num, h in enumerate(headers, 1):
    cell = ws_fixtures.cell(row=4, column=c_num, value=h)
    cell.font, cell.fill, cell.alignment = FONT_HEADER, HEADER_FILL, Alignment(horizontal="center", vertical="center")

for idx, r_data in enumerate(processed_results, start=5):
    row_vals = [r_data["id"], r_data["category"], r_data["text"], r_data["char_count"], r_data["min_expected"], r_data["max_expected"], r_data["actual_segments"], r_data["status"], r_data["latency_ms"], r_data["description"]]
    for c_num, val in enumerate(row_vals, 1):
        cell = ws_fixtures.cell(row=idx, column=c_num, value=val)
        cell.font, cell.fill, cell.border = FONT_BODY, (ZEBRA_FILL if idx % 2 == 0 else WHITE_FILL), THIN_BORDER
        if c_num in [1, 4, 5, 6, 7, 8]: cell.alignment = Alignment(horizontal="center", vertical="center")
        if c_num == 8: cell.font, cell.fill = (FONT_PASS, PASS_FILL) if val == "PASSED" else (FONT_FAIL, FAIL_FILL)

# Sheet 3: Metadata Target Mapping (Task 340 View)
ws_segments = wb.create_sheet(title="Extracted Schema Segments")
ws_segments.views.sheetView[0].showGridLines = True
seg_hdrs = ["Parent Story ID", "Segment ID", "Extracted Segment Text Mapping", "Assigned Language Tag", "Language Confidence", "Start Char Position", "End Char Position"]

for c_num, h in enumerate(seg_hdrs, 1):
    cell = ws_segments.cell(row=4, column=c_num, value=h)
    cell.font, cell.fill, cell.alignment = FONT_HEADER, HEADER_FILL, Alignment(horizontal="center", vertical="center")

s_idx = 5
for r_data in processed_results:
    for s in r_data["segments_data"]:
        vals = [r_data["id"], s["segment_id"], s["text"], s["lang"], s.get("lang_confidence", "regex"), s["start_char"], s["end_char"]]
        for c_num, val in enumerate(vals, 1):
            cell = ws_segments.cell(row=s_idx, column=c_num, value=val)
            cell.font, cell.fill, cell.border = FONT_BODY, (ZEBRA_FILL if s_idx % 2 == 0 else WHITE_FILL), THIN_BORDER
            if c_num in [1, 2, 4, 5, 6]: cell.alignment = Alignment(horizontal="center", vertical="center")
        s_idx += 1

# Normalize Widths
for ws in [ws_dash, ws_fixtures, ws_segments]:
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(len(str(cell.value or '')) for cell in col)
        if col_letter in ['B', 'C'] and ws.title == "Sprint Dashboard": ws.column_dimensions[col_letter].width = 42
        else: ws.column_dimensions[col_letter].width = min(max(max_len + 4, 11), 60)

report_path = os.path.join(MOUNTED_FOLDER, os.getenv("SPLITTER_REPORT_PATH", "Hybrid_test_report.xlsx"))
wb.save(report_path)
logger.info("Profile processed successfully at %s Standard Deviations", SIGMA_MULTIPLIER)
logger.info("Spreadsheet updated: %s", report_path)
