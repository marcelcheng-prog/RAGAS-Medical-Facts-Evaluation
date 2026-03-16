import argparse
import json
import os
from datetime import datetime, timezone


def format_patient_measurement(m):
    parts = []
    if m.get("parameter"):
        parts.append(m.get("parameter"))
    if m.get("value"):
        parts.append(str(m.get("value")))
    if m.get("unit"):
        parts.append(m.get("unit"))
    if m.get("location"):
        parts.append(f"({m.get('location')})")
    return " ".join(parts)


def format_vital(v):
    parts = [v.get("parameter", "").strip()]
    if v.get("value"):
        parts.append(str(v.get("value")))
    if v.get("unit"):
        parts.append(v.get("unit"))
    if v.get("source"):
        parts.append(f"({v.get('source')})")
    return " ".join([p for p in parts if p])


def join_med_list(meds):
    items = []
    for m in meds:
        name = m.get("name", "").strip()
        dose = m.get("dose") or m.get("dosage") or ""
        freq = m.get("frequency") or ""
        parts = [name]
        if dose:
            parts.append(dose)
        if freq:
            parts.append(freq)
        items.append(" ".join([p for p in parts if p]))
    return "; ".join(items)


def load_source_payload(raw):
    # Supports both plain extractor JSON and test-case envelopes with ground_truth.
    if isinstance(raw, dict) and isinstance(raw.get("ground_truth"), dict):
        return raw["ground_truth"]
    return raw


def list_input_files(input_json, input_dir):
    if input_json:
        return [input_json]
    if not input_dir:
        return []

    files = []
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".json"):
            continue
        if name == "schema.json":
            continue
        files.append(os.path.join(input_dir, name))
    return files


def build_soap_sections(source):
    S = []
    chief = source.get("chief_complaint") or ""
    if chief:
        S.append(f"Hauptgrund: {chief}")

    symptoms = source.get("symptoms", []) or []
    if symptoms:
        for s in symptoms[:3]:
            S.append(s)

    patient_measurements = source.get("patient_measurements", []) or []
    for pm in patient_measurements[:2]:
        S.append(format_patient_measurement(pm))

    # O
    O = []
    vitals = source.get("vital_measurements", []) or source.get("vital_signs", []) or []
    for v in vitals:
        O.append(format_vital(v))

    physical = source.get("physical_examination", []) or []
    for p in physical[:3]:
        O.append(p)

    medications_taken = source.get("medications_taken", []) or []
    if medications_taken:
        meds_line = join_med_list(medications_taken)
        O.append(f"Medikamente aktuell: {meds_line}")

    # A
    A = []
    diag = source.get("diagnostic_hypotheses", []) or []
    if diag:
        for d in diag:
            A.append(d)
    else:
        # conservative impression: list symptoms/vitals without inventing diagnosis
        impressions = []
        if vitals:
            impressions.append("Objektive Messungen vorliegend")
        if symptoms:
            impressions.append("Symptomatik: " + ", ".join(symptoms[:3]))
        if impressions:
            A.append("; ".join(impressions))
        else:
            A.append("Keine expliziten diagnostischen Hypothesen dokumentiert")

    # P
    P = []
    meds_planned = source.get("medications_planned", []) or []
    for m in meds_planned:
        name = m.get("name", "")
        action = m.get("action", "")
        dose = m.get("dose") or m.get("dosage") or ""
        freq = m.get("frequency") or ""
        reason = m.get("reason") or m.get("reasoning") or ""
        parts = [f"Medikament: {name}", f"Aktion: {action}"]
        if dose:
            parts.append(f"Dosis: {dose}")
        if freq:
            parts.append(f"Frequenz: {freq}")
        if reason:
            parts.append(f"Grund: {reason}")
        P.append("; ".join(parts))

    diag_plans = source.get("diagnostic_plans", []) or []
    for d in diag_plans:
        P.append(f"Diagnostik: {d}")

    interventions = source.get("therapeutic_interventions", []) or []
    for t in interventions:
        P.append(t)

    fup = source.get("follow_up_instructions") or {}
    if isinstance(fup, dict):
        na = fup.get("next_appointment")
        monitoring = fup.get("monitoring") or []
        if na:
            P.append(f"Nachkontrolle: {na}")
        for m in monitoring:
            P.append(f"Monitoring: {m}")

    patient_edu = source.get("patient_education", []) or []
    for e in patient_edu:
        P.append(e)

    return {"S": S, "O": O, "A": A, "P": P}


def soap_text_from_sections(secs):
    parts = []
    parts.append("S:")
    if secs["S"]:
        for s in secs["S"]:
            parts.append(f"- {s}")
    else:
        parts.append("- Anamnese: keine Daten")

    parts.append("\nO:")
    if secs["O"]:
        for o in secs["O"]:
            parts.append(f"- {o}")
    else:
        parts.append("- keine objektiven Befunde dokumentiert")

    parts.append("\nA:")
    if secs["A"]:
        for a in secs["A"]:
            parts.append(f"- {a}")
    else:
        parts.append("- keine Beurteilung vorhanden")

    parts.append("\nP:")
    if secs["P"]:
        for p in secs["P"]:
            parts.append(f"- {p}")
    else:
        parts.append("- kein Plan dokumentiert")

    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=False)
    p.add_argument("--input-dir", required=False)
    p.add_argument("--output-json", required=False)
    p.add_argument("--output-txt", required=False)
    p.add_argument("--output-dir", required=False)
    args = p.parse_args()

    if not args.input_json and not args.input_dir:
        p.error("One of --input-json or --input-dir is required")

    input_files = list_input_files(args.input_json, args.input_dir)
    if not input_files:
        p.error("No input JSON files found")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        src = load_source_payload(raw)

        secs = build_soap_sections(src)
        soap_text = soap_text_from_sections(secs)

        out = {
            "metadata": {
                "generated_by": "soap_formatter.py",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_file": input_file
            },
            "soap_text": soap_text,
            "soap": secs,
            "source_extract": src,
            "provenance": {"status": "auto-generated"}
        }

        if args.output_dir:
            base = os.path.splitext(os.path.basename(input_file))[0]
            out_json = os.path.join(args.output_dir, base + ".soap.json")
            out_txt = os.path.join(args.output_dir, base + ".soap.txt") if args.output_txt else None
        else:
            out_json = args.output_json or os.path.splitext(input_file)[0] + ".soap.json"
            out_txt = args.output_txt

        out_dir = os.path.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_json}")

        if out_txt:
            out_txt_dir = os.path.dirname(out_txt)
            if out_txt_dir:
                os.makedirs(out_txt_dir, exist_ok=True)
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(soap_text)
            print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
