"""
Streamlit Web UI for Medical Facts & SOAP Evaluation.

Launch: streamlit run app.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

# Ensure package is importable
import sys
sys.path.insert(0, str(Path(__file__).parent))

from medical_facts_evaluation.config.settings import get_settings
from medical_facts_evaluation.config.thresholds import PRODUCTION, DEVELOPMENT
from medical_facts_evaluation.models.loader import load_test_case, load_all_test_cases
from medical_facts_evaluation.clients.medical_facts import MedicalFactsClient
from medical_facts_evaluation.clients.ragas_client import RagasEvaluator
from medical_facts_evaluation.evaluators.soap_evaluator import (
    evaluate_soap_output,
    load_gold_sections,
)
from medical_facts_evaluation.evaluator import MedicalFactsEvaluator
from medical_facts_evaluation.reporters.json_reporter import JsonReporter
from medical_facts_evaluation.formatters.soap_formatter import build_soap_sections, soap_text_from_sections

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FAVORITES_PATH = DATA_DIR / "favorites.json"
RANKINGS_PATH = DATA_DIR / "rankings.json"
FEEDBACK_DIR = DATA_DIR / "feedback"
FEEDBACK_DIR.mkdir(exist_ok=True)
GOLD_REVIEW_DIR = DATA_DIR / "gold_review"
GOLD_REVIEW_DIR.mkdir(exist_ok=True)
TEST_CASES_DIR = Path(__file__).parent / "medical_facts_evaluation" / "test_cases"
GOLD_SOAP_DIR = Path(__file__).parent / "medical_facts_evaluation" / "gold_soap"
RESULTS_DIR = Path(__file__).parent / "results" / "medical_facts_production"

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_favorites() -> list[dict]:
    return _load_json(FAVORITES_PATH)


def save_favorites(favs: list[dict]) -> None:
    _save_json(FAVORITES_PATH, favs)


def load_rankings() -> list[dict]:
    return _load_json(RANKINGS_PATH)


def save_rankings(rows: list[dict]) -> None:
    _save_json(RANKINGS_PATH, rows)


def add_ranking_entry(entry: dict) -> None:
    rows = load_rankings()
    rows.append(entry)
    save_rankings(rows)


# ---------------------------------------------------------------------------
# Feedback helpers
# ---------------------------------------------------------------------------

def _feedback_path(test_case_id: str, agent_id: str) -> Path:
    safe_tc = test_case_id.replace("/", "_").replace("\\", "_")
    safe_agent = agent_id[:12]
    return FEEDBACK_DIR / f"{safe_tc}_{safe_agent}.json"


def load_feedback(test_case_id: str, agent_id: str) -> list[dict]:
    p = _feedback_path(test_case_id, agent_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def save_feedback_entry(test_case_id: str, agent_id: str, entry: dict) -> None:
    entries = load_feedback(test_case_id, agent_id)
    entries.append(entry)
    p = _feedback_path(test_case_id, agent_id)
    p.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def load_all_feedback() -> list[dict]:
    """Load all feedback entries across all files."""
    all_entries = []
    for p in sorted(FEEDBACK_DIR.glob("*.json")):
        try:
            entries = json.loads(p.read_text(encoding="utf-8"))
            all_entries.extend(entries)
        except Exception:
            pass
    return all_entries


# ---------------------------------------------------------------------------
# Gold-standard review feedback helpers
# ---------------------------------------------------------------------------

def _gold_review_path(test_case_id: str) -> Path:
    safe = test_case_id.replace("/", "_").replace("\\", "_")
    return GOLD_REVIEW_DIR / f"{safe}.json"


def load_gold_review(test_case_id: str) -> list[dict]:
    p = _gold_review_path(test_case_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def save_gold_review_entry(test_case_id: str, entry: dict) -> None:
    entries = load_gold_review(test_case_id)
    entries.append(entry)
    p = _gold_review_path(test_case_id)
    p.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def load_all_gold_reviews() -> list[dict]:
    all_entries = []
    for p in sorted(GOLD_REVIEW_DIR.glob("*.json")):
        try:
            all_entries.extend(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return all_entries


def generate_recommendations(
    mode: str,
    result_data: dict,
    ground_truth_dict: dict | None = None,
    gold_sections: dict | None = None,
) -> list[str]:
    """Auto-generate improvement recommendations from evaluation gaps."""
    recs: list[str] = []

    if mode == "soap":
        eff = result_data.get("effective_section_scores") or result_data.get("section_scores", {})
        pred_sections = result_data.get("soap_sections", {})
        gold = result_data.get("gold_sections", gold_sections or {})

        for sec in ("S", "O", "A", "P"):
            score = eff.get(sec, 0)
            pred_items = pred_sections.get(sec, [])
            gold_items = gold.get(sec, [])
            pred_norm = {p.strip().lower() for p in pred_items if p.strip()}
            gold_norm = {g.strip().lower() for g in gold_items if g.strip()}
            missing = [g for g in gold_items if g.strip().lower() not in pred_norm]
            extra = [p for p in pred_items if p.strip().lower() not in gold_norm]

            if not pred_items and gold_items:
                recs.append(f"🔴 Section {sec} is completely empty — agent is not producing any {sec} content. Add explicit instructions to extract {sec} items.")
            elif score < 0.4:
                recs.append(f"🔴 Section {sec} scores very low ({score:.0%}) — major prompt rewrite needed for this section.")
            elif score < 0.6:
                recs.append(f"🟡 Section {sec} is weak ({score:.0%}) — review missing items and add examples to prompt.")

            if missing:
                missing_preview = "; ".join(missing[:3])
                if len(missing) > 3:
                    missing_preview += f" (+{len(missing)-3} more)"
                recs.append(f"📋 {sec}: {len(missing)} missing gold items — e.g. {missing_preview}")

            if len(extra) > len(gold_items) and len(extra) > 3:
                recs.append(f"⚠️ {sec}: {len(extra)} extra items vs {len(gold_items)} gold — agent may be over-generating. Tighten extraction scope.")

        overall = result_data.get("overall_score", 0)
        if overall < 50:
            recs.append("🔴 Overall score < 50 — fundamental prompt/agent architecture issue. Consider restructuring the agent graph.")
        elif overall < 70:
            recs.append("🟡 Overall score < 70 — targeted improvements needed. Focus on weakest sections first.")

        lex_avg = result_data.get("lexical_average_content_score", 0)
        sem_avg = result_data.get("semantic_average_content_score")
        if sem_avg is not None and sem_avg > 0.7 and lex_avg < 0.4:
            recs.append("💡 High semantic but low lexical — agent captures intent but uses different wording. Align output style with gold schema (e.g. '1-0-0' vs 'morgens').")

    elif mode == "medical_facts":
        quality = result_data.get("quality_score", 0)
        med_eval = result_data.get("medication_eval", {})

        if isinstance(med_eval, dict):
            precision = med_eval.get("name_precision", 1.0)
            recall = med_eval.get("name_recall", 1.0)
            hallucinations = med_eval.get("hallucinations", [])
            missing_meds = med_eval.get("missing_medications", [])

            if hallucinations:
                recs.append(f"🔴 CRITICAL: {len(hallucinations)} hallucinated medications: {', '.join(hallucinations[:3])}. Add stricter transcript-only extraction rules.")
            if precision < 0.9:
                recs.append(f"🟡 Medication precision low ({precision:.0%}) — agent is extracting incorrect medications.")
            if recall < 0.8:
                recs.append(f"🟡 Medication recall low ({recall:.0%}) — agent is missing medications.")
            if missing_meds:
                recs.append(f"📋 Missing medications: {', '.join(missing_meds[:5])}")

        faithfulness = result_data.get("faithfulness")
        if faithfulness is not None and faithfulness < 0.85:
            recs.append(f"🟡 Faithfulness low ({faithfulness:.0%}) — agent is generating content not supported by transcript.")

        context_recall = result_data.get("context_recall")
        if context_recall is not None and context_recall < 0.8:
            recs.append(f"🟡 Context recall low ({context_recall:.0%}) — agent is missing important transcript information.")

        if quality < 60:
            recs.append("🔴 Quality score < 60 — fundamental issues. Review agent prompt and extraction logic.")

    if not recs:
        recs.append("✅ No major issues detected. Agent is performing well on this test case.")

    return recs


# ---------------------------------------------------------------------------
# Test case discovery
# ---------------------------------------------------------------------------

def discover_test_cases() -> dict[str, Path]:
    cases: dict[str, Path] = {}
    if TEST_CASES_DIR.exists():
        for p in sorted(TEST_CASES_DIR.glob("*.json")):
            if p.name == "schema.json":
                continue
            cases[p.stem] = p
    return cases


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Facts & SOAP Evaluation",
    page_icon="🏥",
    layout="wide",
)

# Known / predefined agents (label → ID)
KNOWN_AGENTS: dict[str, str] = {
    "DEV-FLH agent": "ff04c2b01edb11f194964348756e437e",
    "medicalfactssoapv2 agent": "f44320e61ef011f194964348756e437e",
}


# ---------------------------------------------------------------------------
# Sidebar: Settings & Favorites
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

settings = get_settings()
validation_errors = settings.validate()
if validation_errors:
    for e in validation_errors:
        st.sidebar.error(e)
    st.sidebar.info("Configure your .env file with API keys.")

# Mode selector
mode = st.sidebar.radio("Evaluation Mode", ["Medical Facts", "SOAP"], index=0)

# Agent input
st.sidebar.subheader("Agent Configuration")
favorites = load_favorites()
fav_labels = [f"{f['label']} ({f['agent_id'][:8]}…)" for f in favorites]

# Build agent choices: known agents + favorites + custom
_agent_options: list[str] = list(KNOWN_AGENTS.keys())
for f in favorites:
    lbl = f"{f['label']} ⭐"
    if lbl not in _agent_options:
        _agent_options.append(lbl)
_agent_options.append("✏️ Custom Agent ID")

_agent_a_choice = st.sidebar.selectbox("Agent A", _agent_options, index=0, key="agent_a_choice")
if _agent_a_choice == "✏️ Custom Agent ID":
    agent_id = st.sidebar.text_input("Agent ID (A)", value=settings.default_agent_id, key="agent_a_custom")
elif _agent_a_choice in KNOWN_AGENTS:
    agent_id = KNOWN_AGENTS[_agent_a_choice]
    st.sidebar.caption(f"`{agent_id}`")
else:
    # It's a favorite
    _fav_match = next((f for f in favorites if f"{f['label']} ⭐" == _agent_a_choice), None)
    agent_id = _fav_match["agent_id"] if _fav_match else settings.default_agent_id
    st.sidebar.caption(f"`{agent_id}`")

# Compare mode
compare_mode = st.sidebar.checkbox("🆚 Compare two agents", value=False)
agent_b_id = ""
if compare_mode:
    _agent_b_choice = st.sidebar.selectbox("Agent B", _agent_options, index=min(1, len(_agent_options) - 1), key="agent_b_choice")
    if _agent_b_choice == "✏️ Custom Agent ID":
        agent_b_id = st.sidebar.text_input("Agent ID (B)", value="", placeholder="Enter second agent ID", key="agent_b_custom")
    elif _agent_b_choice in KNOWN_AGENTS:
        agent_b_id = KNOWN_AGENTS[_agent_b_choice]
        st.sidebar.caption(f"`{agent_b_id}`")
    else:
        _fav_b_match = next((f for f in favorites if f"{f['label']} ⭐" == _agent_b_choice), None)
        agent_b_id = _fav_b_match["agent_id"] if _fav_b_match else ""
        st.sidebar.caption(f"`{agent_b_id}`")

# Test case selector
test_cases = discover_test_cases()
tc_names = list(test_cases.keys())
all_cases = st.sidebar.checkbox("Run all test cases", value=False)
if not all_cases:
    selected_tc = st.sidebar.selectbox("Test Case", tc_names, index=tc_names.index("michael_mueller") if "michael_mueller" in tc_names else 0)
else:
    selected_tc = None

# Thresholds
threshold_profile = st.sidebar.selectbox("Threshold Profile", ["production", "development"])

# SOAP-specific
if mode == "SOAP":
    soap_structure_thresh = st.sidebar.slider(
        "Structure Threshold", 0.0, 1.0, 1.0, 0.05,
        help="Minimum fraction of SOAP sections (S, O, A, P) that must be present. "
             "1.0 = all 4 required, 0.75 = 3 of 4 suffice. Below this the test fails.",
    )
    soap_content_thresh = st.sidebar.slider(
        "Content Threshold", 0.0, 1.0, 0.70, 0.05,
        help="Minimum average content quality score (weighted blend of lexical, "
             "GPT-semantic, and RAGAS) across all SOAP sections. Below this the test fails.",
    )

# ---------------------------------------------------------------------------
# Sidebar: Favorites Management
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⭐ Manage Favorites")

with st.sidebar.expander("Add Favorite"):
    fav_label = st.text_input("Label", placeholder="e.g. SOAP v2 Best")
    fav_agent_id = st.text_input("Agent ID to save", value=agent_id)
    fav_mode = st.selectbox("Mode", ["medical_facts", "soap"])
    if st.button("Save Favorite"):
        if fav_label and fav_agent_id:
            favorites.append({
                "label": fav_label,
                "agent_id": fav_agent_id,
                "mode": fav_mode,
                "added": datetime.now().isoformat(),
            })
            save_favorites(favorites)
            st.success(f"Saved: {fav_label}")
            st.rerun()

if favorites:
    with st.sidebar.expander("Current Favorites"):
        for i, f in enumerate(favorites):
            col1, col2 = st.columns([3, 1])
            col1.write(f"**{f['label']}** — `{f['agent_id'][:12]}…` ({f['mode']})")
            if col2.button("🗑️", key=f"del_fav_{i}"):
                favorites.pop(i)
                save_favorites(favorites)
                st.rerun()


# ---------------------------------------------------------------------------
# Main Area
# ---------------------------------------------------------------------------
st.title("🏥 Medical Facts & SOAP Evaluation Dashboard")

tab_eval, tab_browse, tab_goldreview, tab_ranking, tab_feedback, tab_favorites = st.tabs(["📊 Evaluate", "📋 Data Browser", "📖 Gold Standard Review", "🏆 Rankings", "💬 Feedback", "⭐ Favorites"])


# ========================== EVALUATE TAB ==========================
with tab_eval:
    st.header(f"{'🧼 SOAP' if mode == 'SOAP' else '💊 Medical Facts'} Evaluation")

    if not agent_id:
        st.warning("Enter an Agent ID in the sidebar to begin.")
        st.stop()
    if compare_mode and not agent_b_id:
        st.warning("Enter Agent B ID in the sidebar for comparison.")
        st.stop()

    run_btn = st.button("🚀 Run Evaluation", type="primary", use_container_width=True)

    if run_btn:
        thresholds = PRODUCTION if threshold_profile == "production" else DEVELOPMENT
        agents_to_run = [("A", agent_id)]
        if compare_mode and agent_b_id:
            agents_to_run.append(("B", agent_b_id))

        # ---- Helper: convert MF EvaluationResult to dict for recommendations ----
        def _mf_to_dict(result) -> dict:
            d = {
                "quality_score": result.quality_score,
                "faithfulness": result.faithfulness,
                "context_recall": result.context_recall,
            }
            if result.medication_eval:
                d["medication_precision"] = result.medication_eval.name_precision
                d["medication_recall"] = result.medication_eval.name_recall
            return d

        # ---- Helper: render feedback form + auto-recommendations ----
        def _render_feedback_block(eval_mode: str, aid: str, tc, result_data: dict, uid: str):
            """Render auto-recommendations and a feedback form inside a result expander."""
            with st.expander("🤖 Auto-Recommendations"):
                recs = generate_recommendations(eval_mode, result_data)
                for r in recs:
                    st.markdown(f"- {r}")

            with st.expander("💬 Leave Feedback"):
                # Show existing feedback
                existing = load_feedback(tc.test_id, aid)
                if existing:
                    st.caption(f"{len(existing)} previous feedback entries")
                    for fb in existing[-3:]:
                        st.markdown(f"> **{fb.get('timestamp', '')[:16]}** — {fb.get('feedback', '')}")
                        if fb.get("recommendations"):
                            st.caption("Auto-recs: " + "; ".join(fb["recommendations"][:2]))
                    if len(existing) > 3:
                        st.caption(f"…and {len(existing)-3} more (see Feedback tab)")

                fb_text = st.text_area(
                    "Your feedback on this agent's output",
                    placeholder="e.g. Section O is missing vital signs, medication format is inconsistent…",
                    key=f"fb_text_{uid}",
                )
                fb_save_recs = st.checkbox("Include auto-recommendations", value=True, key=f"fb_recs_{uid}")
                if st.button("💾 Save Feedback", key=f"fb_save_{uid}"):
                    if fb_text.strip():
                        entry = {
                            "timestamp": datetime.now().isoformat(),
                            "agent_id": aid,
                            "test_case_id": tc.test_id,
                            "test_case_name": tc.name,
                            "mode": eval_mode,
                            "feedback": fb_text.strip(),
                            "recommendations": generate_recommendations(eval_mode, result_data) if fb_save_recs else [],
                            "score": result_data.get("quality_score") or result_data.get("overall_score"),
                        }
                        save_feedback_entry(tc.test_id, aid, entry)
                        st.success("Feedback saved!")
                    else:
                        st.warning("Enter some feedback text first.")

        # ---- Helper: run one medical-facts eval for an agent ----
        def _run_mf_eval(aid: str, tc, thresholds):
            client = MedicalFactsClient.from_settings(aid, settings, verbose=False)
            openai_client = OpenAI(api_key=settings.openai_api_key)
            ragas_eval = RagasEvaluator.from_settings(openai_client, settings, verbose=False)
            evaluator = MedicalFactsEvaluator(
                client=client, ragas=ragas_eval, thresholds=thresholds, verbose=False,
            )
            result = evaluator.evaluate(tc)
            add_ranking_entry({
                "timestamp": datetime.now().isoformat(),
                "agent_id": aid,
                "mode": "medical_facts",
                "test_case": tc.test_id,
                "quality_score": result.quality_score,
                "passed": result.passed,
                "api_time_seconds": result.api_time_seconds,
                "faithfulness": result.faithfulness,
                "context_recall": result.context_recall,
                "medication_precision": result.medication_eval.name_precision if result.medication_eval else None,
                "medication_recall": result.medication_eval.name_recall if result.medication_eval else None,
            })
            return result

        # ---- Helper: run one SOAP eval for an agent ----
        def _run_soap_eval(aid, tc, tc_path, openai_client, ragas_eval):
            client = MedicalFactsClient.from_settings(aid, settings, verbose=False)
            response = client.extract_facts(tc.transcript)
            if response.error:
                return {"error": response.error, "test_name": tc.name, "agent_id": aid}
            gold = load_gold_sections(tc_path, GOLD_SOAP_DIR)
            eval_result = evaluate_soap_output(
                output=response.content,
                gold_sections=gold,
                test_id=tc.test_id,
                test_name=tc.name,
                agent_id=aid,
                api_time_seconds=response.api_time_seconds,
                structure_threshold=soap_structure_thresh if mode == "SOAP" else 1.0,
                content_threshold=soap_content_thresh if mode == "SOAP" else 0.70,
                openai_client=openai_client,
                openai_model=settings.openai_model,
                ragas_evaluator=ragas_eval,
                transcript=tc.transcript,
            )
            rd = eval_result.to_dict()
            add_ranking_entry({
                "timestamp": datetime.now().isoformat(),
                "agent_id": aid,
                "mode": "soap",
                "test_case": tc.test_id,
                "quality_score": rd["overall_score"],
                "passed": rd["passed"],
                "api_time_seconds": rd["api_time_seconds"],
                "structure_score": rd["structure_score"],
                "content_score": rd["average_content_score"],
                "lexical_score": rd.get("lexical_average_content_score"),
                "semantic_gpt_score": rd.get("semantic_average_content_score"),
                "semantic_ragas_score": rd.get("ragas_average_content_score"),
            })
            return rd

        # ---- Helper: render a single MF result ----
        def _render_mf_result(result, tc, key_suffix=""):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Quality Score", f"{result.quality_score:.1f}")
            col2.metric("API Time", f"{result.api_time_seconds:.2f}s")
            col3.metric("Faithfulness", f"{result.faithfulness:.1%}" if result.faithfulness else "N/A")
            col4.metric("Context Recall", f"{result.context_recall:.1%}" if result.context_recall else "N/A")

            if result.medication_eval:
                st.markdown("**Medication Metrics**")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Precision", f"{result.medication_eval.name_precision:.1%}")
                mc2.metric("Recall", f"{result.medication_eval.name_recall:.1%}")
                mc3.metric("F1", f"{result.medication_eval.f1_score:.1%}")
                mc4.metric("Hallucinations", str(len(result.medication_eval.hallucinations)))
                if result.medication_eval.missing_medications:
                    st.warning(f"Missing: {', '.join(result.medication_eval.missing_medications)}")
                if result.medication_eval.hallucinations:
                    st.error(f"Hallucinations: {', '.join(result.medication_eval.hallucinations)}")
            if result.failure_reasons:
                st.error("Failures: " + "; ".join(result.failure_reasons))
            with st.expander("Raw Agent Output"):
                st.code(result.agent_output_raw[:5000] if result.agent_output_raw else "N/A", language="json")

        # ---- Helper: render a single SOAP result ----
        def _render_soap_result(rd, tc, key_suffix=""):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall", f"{rd['overall_score']:.1f}/100")
            col2.metric("Structure", f"{rd['structure_score']:.0%}")
            col3.metric("Content", f"{rd['average_content_score']:.1%}")
            col4.metric("API Time", f"{rd['api_time_seconds']:.2f}s")

            st.markdown("**Section Scores**")
            scol1, scol2, scol3, scol4 = st.columns(4)
            eff = rd.get("effective_section_scores") or rd.get("section_scores", {})
            lex = rd.get("section_scores", {})
            for col, sec in zip([scol1, scol2, scol3, scol4], ["S", "O", "A", "P"]):
                col.metric(f"{sec} (eff)", f"{eff.get(sec, 0):.1%}", delta=f"lex: {lex.get(sec, 0):.1%}", delta_color="off")

            st.markdown("**Scoring Layers**")
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("Lexical", f"{rd.get('lexical_average_content_score', 0):.1%}")
            sem_gpt = rd.get("semantic_average_content_score")
            lc2.metric("GPT Semantic", f"{sem_gpt:.1%}" if sem_gpt is not None else "N/A")
            sem_ragas = rd.get("ragas_average_content_score")
            lc3.metric("RAGAS Semantic", f"{sem_ragas:.1%}" if sem_ragas is not None else "N/A")

            for sec in ("S", "O", "A", "P"):
                pred = rd.get("soap_sections", {}).get(sec, [])
                gold_items = rd.get("gold_sections", {}).get(sec, [])
                with st.expander(f"Section {sec} — {len(pred)} pred / {len(gold_items)} gold"):
                    pc, gc = st.columns(2)
                    with pc:
                        st.markdown("**Predicted:**")
                        for item in (pred or ["_(empty)_"]):
                            st.markdown(f"- {item}")
                    with gc:
                        st.markdown("**Gold:**")
                        for item in (gold_items or ["_(empty)_"]):
                            st.markdown(f"- {item}")
                    pred_norm = {p.strip().lower() for p in pred if p.strip()}
                    gold_norm = {g.strip().lower() for g in gold_items if g.strip()}
                    missing = [g for g in gold_items if g.strip().lower() not in pred_norm]
                    extra = [p for p in pred if p.strip().lower() not in gold_norm]
                    if missing:
                        st.warning(f"Missing: {', '.join(missing)}")
                    if extra:
                        st.info(f"Extra: {', '.join(extra)}")

            if rd.get("failure_reasons"):
                st.error("Failures: " + "; ".join(rd["failure_reasons"]))
            if rd.get("warnings"):
                st.warning("Warnings: " + "; ".join(rd["warnings"]))
            with st.expander("Raw Agent Output"):
                st.code(rd.get("raw_output", "")[:5000], language="json")

        # ==================================================================
        # MEDICAL FACTS MODE
        # ==================================================================
        if mode == "Medical Facts":
            cases_to_run = list(test_cases.values()) if all_cases else [test_cases[selected_tc]]

            progress = st.progress(0, text="Starting evaluation…")
            # Collect results per test case: list of (tc, {label: result})
            case_results: list[tuple] = []
            total_steps = len(cases_to_run) * len(agents_to_run)
            step = 0
            for tc_path in cases_to_run:
                tc = load_test_case(tc_path)
                agent_results = {}
                for label, aid in agents_to_run:
                    step += 1
                    progress.progress(step / total_steps, text=f"Agent {label}: {tc.name}")
                    with st.spinner(f"Agent {label} ({aid[:8]}…) → {tc.name}"):
                        agent_results[label] = _run_mf_eval(aid, tc, thresholds)
                case_results.append((tc, agent_results))
            progress.progress(1.0, text="Done!")

            # --- Comparison summary table ---
            if compare_mode:
                st.subheader("🆚 Comparison Summary")
                summary_rows = []
                wins_a, wins_b = 0, 0
                for tc, ares in case_results:
                    ra, rb = ares.get("A"), ares.get("B")
                    if ra and rb:
                        winner = "A" if ra.quality_score > rb.quality_score else ("B" if rb.quality_score > ra.quality_score else "Tie")
                        if winner == "A":
                            wins_a += 1
                        elif winner == "B":
                            wins_b += 1
                        summary_rows.append({
                            "Test Case": tc.name[:40],
                            f"Agent A ({agent_id[:8]}…)": f"{ra.quality_score:.1f}",
                            f"Agent B ({agent_b_id[:8]}…)": f"{rb.quality_score:.1f}",
                            "⏱ A": f"{ra.api_time_seconds:.1f}s",
                            "⏱ B": f"{rb.api_time_seconds:.1f}s",
                            "Winner": f"{'🅰️' if winner == 'A' else '🅱️' if winner == 'B' else '🤝'} {winner}",
                        })
                st.dataframe(summary_rows, use_container_width=True, hide_index=True)
                st.info(f"**Overall: Agent A wins {wins_a}, Agent B wins {wins_b}, Ties {len(case_results) - wins_a - wins_b}**")

            # --- Per test case detail ---
            for tc, ares in case_results:
                if compare_mode:
                    ra, rb = ares.get("A"), ares.get("B")
                    score_a = ra.quality_score if ra else 0
                    score_b = rb.quality_score if rb else 0
                    header = f"{tc.name} — A: {score_a:.1f} vs B: {score_b:.1f}"
                    with st.expander(header, expanded=False):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"### 🅰️ Agent A (`{agent_id[:12]}…`)")
                            if ra:
                                _render_mf_result(ra, tc, key_suffix="_a")
                                _render_feedback_block("medical_facts", agent_id, tc, _mf_to_dict(ra), f"mf_cmp_a_{tc.test_id}")
                        with col_b:
                            st.markdown(f"### 🅱️ Agent B (`{agent_b_id[:12]}…`)")
                            if rb:
                                _render_mf_result(rb, tc, key_suffix="_b")
                                _render_feedback_block("medical_facts", agent_b_id, tc, _mf_to_dict(rb), f"mf_cmp_b_{tc.test_id}")
                        with st.expander("📝 Ground Truth"):
                            st.json(tc.ground_truth.to_dict())
                        with st.expander("🎙️ Transcript"):
                            st.text_area("Transcript", value=tc.transcript, height=300, disabled=True, key=f"mf_cmp_tr_{tc.test_id}")
                else:
                    result = ares["A"]
                    status = "✅ PASSED" if result.passed else "❌ FAILED"
                    with st.expander(f"{status} — {result.test_name} — Score: {result.quality_score:.1f}/100", expanded=not result.passed):
                        _render_mf_result(result, tc)
                        with st.expander("📝 Ground Truth"):
                            st.json(tc.ground_truth.to_dict())
                        with st.expander("🎙️ Transcript"):
                            st.text_area("Transcript", value=tc.transcript, height=300, disabled=True, key=f"mf_transcript_{tc.test_id}")
                        _render_feedback_block("medical_facts", agent_id, tc, _mf_to_dict(result), f"mf_{tc.test_id}")

        # ==================================================================
        # SOAP MODE
        # ==================================================================
        else:
            cases_to_run = list(test_cases.values()) if all_cases else [test_cases[selected_tc]]
            openai_client = OpenAI(api_key=settings.openai_api_key)
            ragas_eval = RagasEvaluator.from_settings(openai_client, settings, verbose=False)

            progress = st.progress(0, text="Starting SOAP evaluation…")
            case_results: list[tuple] = []
            total_steps = len(cases_to_run) * len(agents_to_run)
            step = 0
            for tc_path in cases_to_run:
                tc = load_test_case(tc_path)
                agent_results = {}
                for label, aid in agents_to_run:
                    step += 1
                    progress.progress(step / total_steps, text=f"SOAP Agent {label}: {tc.name}")
                    with st.spinner(f"SOAP Agent {label} ({aid[:8]}…) → {tc.name}"):
                        rd = _run_soap_eval(aid, tc, tc_path, openai_client, ragas_eval)
                        if "error" in rd:
                            st.error(f"API Error ({label}): {rd['error']}")
                        agent_results[label] = rd
                case_results.append((tc, tc_path, agent_results))
            progress.progress(1.0, text="Done!")

            # --- Comparison summary table ---
            if compare_mode:
                st.subheader("🆚 SOAP Comparison Summary")
                summary_rows = []
                wins_a, wins_b = 0, 0
                for tc, tc_path, ares in case_results:
                    ra, rb = ares.get("A", {}), ares.get("B", {})
                    sa = ra.get("overall_score", 0) if isinstance(ra, dict) else 0
                    sb = rb.get("overall_score", 0) if isinstance(rb, dict) else 0
                    winner = "A" if sa > sb else ("B" if sb > sa else "Tie")
                    if winner == "A":
                        wins_a += 1
                    elif winner == "B":
                        wins_b += 1
                    summary_rows.append({
                        "Test Case": tc.name[:40],
                        f"Agent A ({agent_id[:8]}…)": f"{sa:.1f}",
                        f"Agent B ({agent_b_id[:8]}…)": f"{sb:.1f}",
                        "⏱ A": f"{ra.get('api_time_seconds', 0):.1f}s" if isinstance(ra, dict) else "—",
                        "⏱ B": f"{rb.get('api_time_seconds', 0):.1f}s" if isinstance(rb, dict) else "—",
                        "Winner": f"{'🅰️' if winner == 'A' else '🅱️' if winner == 'B' else '🤝'} {winner}",
                    })
                st.dataframe(summary_rows, use_container_width=True, hide_index=True)
                st.info(f"**Overall: Agent A wins {wins_a}, Agent B wins {wins_b}, Ties {len(case_results) - wins_a - wins_b}**")

            # --- Per test case detail ---
            for tc, tc_path, ares in case_results:
                if compare_mode:
                    ra, rb = ares.get("A", {}), ares.get("B", {})
                    sa = ra.get("overall_score", 0) if isinstance(ra, dict) and "error" not in ra else 0
                    sb = rb.get("overall_score", 0) if isinstance(rb, dict) and "error" not in rb else 0
                    header = f"{tc.name} — A: {sa:.1f} vs B: {sb:.1f}"
                    with st.expander(header, expanded=False):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"### 🅰️ Agent A (`{agent_id[:12]}…`)")
                            if isinstance(ra, dict) and "error" not in ra:
                                _render_soap_result(ra, tc)
                            else:
                                st.error(f"Error: {ra.get('error', 'unknown')}")
                        with col_b:
                            st.markdown(f"### 🅱️ Agent B (`{agent_b_id[:12]}…`)")
                            if isinstance(rb, dict) and "error" not in rb:
                                _render_soap_result(rb, tc)
                            else:
                                st.error(f"Error: {rb.get('error', 'unknown')}")
                        with st.expander("🧼 Gold SOAP Standard"):
                            gold_for_display = ra.get("gold_sections", {}) if isinstance(ra, dict) else {}
                            for sec in ("S", "O", "A", "P"):
                                items = gold_for_display.get(sec, [])
                                st.markdown(f"**{sec}** ({len(items)} items)")
                                for item in items:
                                    st.markdown(f"- {item}")
                        with st.expander("📝 Ground Truth"):
                            st.json(tc.ground_truth.to_dict())
                        with st.expander("🎙️ Transcript"):
                            st.text_area("Transcript", value=tc.transcript, height=300, disabled=True, key=f"soap_cmp_tr_{tc.test_id}")
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            if isinstance(ra, dict) and "error" not in ra:
                                _render_feedback_block("soap", agent_id, tc, ra, f"soap_cmp_a_{tc.test_id}")
                        with sc2:
                            if isinstance(rb, dict) and "error" not in rb:
                                _render_feedback_block("soap", agent_b_id, tc, rb, f"soap_cmp_b_{tc.test_id}")
                else:
                    rd = ares.get("A", {})
                    if isinstance(rd, dict) and "error" in rd:
                        st.error(f"API Error for {tc.name}: {rd['error']}")
                        continue
                    status = "✅ PASSED" if rd.get("passed") else "❌ FAILED"
                    with st.expander(f"{status} — {rd.get('test_name', tc.name)} — Score: {rd.get('overall_score', 0):.1f}/100", expanded=not rd.get("passed", False)):
                        _render_soap_result(rd, tc)
                        with st.expander("🧼 Gold SOAP Standard"):
                            gold_for_display = rd.get("gold_sections", {})
                            for sec in ("S", "O", "A", "P"):
                                items = gold_for_display.get(sec, [])
                                st.markdown(f"**{sec}** ({len(items)} items)")
                                for item in items:
                                    st.markdown(f"- {item}")
                        with st.expander("📝 Ground Truth (Medical Facts)"):
                            st.json(tc.ground_truth.to_dict())
                        with st.expander("🎙️ Transcript"):
                            st.text_area("Transcript", value=tc.transcript, height=300, disabled=True, key=f"soap_transcript_{tc.test_id}")
                        _render_feedback_block("soap", agent_id, tc, rd, f"soap_{tc.test_id}")


# ========================== DATA BROWSER TAB ==========================
with tab_browse:
    st.header("📋 Data Browser")
    st.caption("Browse transcripts, ground truth, gold SOAP standards, and past evaluation outputs.")

    browse_tc_names = list(test_cases.keys())
    browse_selected = st.selectbox(
        "Select Test Case",
        browse_tc_names,
        index=browse_tc_names.index("michael_mueller") if "michael_mueller" in browse_tc_names else 0,
        key="browse_tc",
    )

    if browse_selected:
        browse_path = test_cases[browse_selected]
        browse_tc = load_test_case(browse_path)

        # --- Transcript ---
        st.subheader("🎙️ Transcript")
        edited_transcript = st.text_area(
            "Transcript (editable — changes are used for this session only)",
            value=browse_tc.transcript,
            height=400,
            key=f"browse_transcript_{browse_selected}",
        )
        st.download_button(
            "📥 Download Transcript",
            data=edited_transcript,
            file_name=f"{browse_selected}_transcript.txt",
            mime="text/plain",
            key=f"dl_transcript_{browse_selected}",
        )

        # --- Ground Truth (Medical Facts) ---
        st.subheader("📝 Ground Truth (Medical Facts)")
        gt = browse_tc.ground_truth
        gt_dict = gt.to_dict()

        gt_tab_summary, gt_tab_json = st.tabs(["Summary", "Raw JSON"])

        with gt_tab_summary:
            # Medications taken
            if gt.medications_taken:
                st.markdown("**Medications Taken (Bestandsmedikation):**")
                for m in gt.medications_taken:
                    parts = [f"**{m.name}**"]
                    if m.dose:
                        parts.append(m.dose)
                    if m.frequency:
                        parts.append(m.frequency)
                    if m.action:
                        parts.append(f"({m.action})")
                    st.markdown("- " + " — ".join(parts))

            # Medications planned
            if gt.medications_planned:
                st.markdown("**Medications Planned (Geplante Medikation):**")
                for m in gt.medications_planned:
                    parts = [f"**{m.name}**"]
                    if m.dose:
                        parts.append(m.dose)
                    if m.frequency:
                        parts.append(m.frequency)
                    if m.action:
                        parts.append(f"({m.action})")
                    if m.reason:
                        parts.append(f"— {m.reason}")
                    st.markdown("- " + " — ".join(parts))

            # All medication names
            if gt.all_medication_names:
                st.markdown(f"**All Medication Names:** {', '.join(gt.all_medication_names)}")

            # Vital measurements
            if gt.vital_measurements:
                st.markdown("**Vital Measurements:**")
                for v in gt.vital_measurements:
                    st.markdown(f"- {v.parameter}: {v.value} {v.unit} ({v.source})")

            # Symptoms
            if gt.symptoms:
                st.markdown(f"**Symptoms:** {', '.join(gt.symptoms)}")

            # Medical history
            if gt.medical_history:
                st.markdown(f"**Medical History:** {', '.join(gt.medical_history)}")

            # Diagnostic plans
            if gt.diagnostic_plans:
                st.markdown(f"**Diagnostic Plans:** {', '.join(gt.diagnostic_plans)}")

            # Therapeutic interventions
            if gt.therapeutic_interventions:
                st.markdown(f"**Therapeutic Interventions:** {', '.join(gt.therapeutic_interventions)}")

            # Forbidden medications
            if gt.forbidden_medications:
                st.error(f"**Forbidden Medications:** {', '.join(gt.forbidden_medications)}")

        with gt_tab_json:
            st.json(gt_dict)
            st.download_button(
                "📥 Download Ground Truth JSON",
                data=json.dumps(gt_dict, ensure_ascii=False, indent=2),
                file_name=f"{browse_selected}_ground_truth.json",
                mime="application/json",
                key="dl_gt",
            )

        # --- Gold SOAP ---
        st.subheader("🧼 Gold Standard SOAP")
        gold_json_path = GOLD_SOAP_DIR / f"{browse_selected}.soap.json"
        gold_txt_path = GOLD_SOAP_DIR / f"{browse_selected}.soap.txt"

        if gold_json_path.exists():
            gold_data = json.loads(gold_json_path.read_text(encoding="utf-8"))
            soap_sections = gold_data.get("soap", {})

            soap_tab_formatted, soap_tab_text, soap_tab_json = st.tabs(["Formatted", "Plain Text", "Raw JSON"])

            with soap_tab_formatted:
                for sec_label, sec_key in [("S — Subjective", "S"), ("O — Objective", "O"), ("A — Assessment", "A"), ("P — Plan", "P")]:
                    items = soap_sections.get(sec_key, [])
                    st.markdown(f"**{sec_label}** ({len(items)} items)")
                    if items:
                        for item in items:
                            st.markdown(f"- {item}")
                    else:
                        st.caption("(empty)")

            with soap_tab_text:
                soap_text = gold_data.get("soap_text", "")
                if not soap_text and gold_txt_path.exists():
                    soap_text = gold_txt_path.read_text(encoding="utf-8")
                st.text_area("SOAP Text", value=soap_text, height=300, disabled=True, key="gold_soap_text")

            with soap_tab_json:
                st.json(gold_data)
                st.download_button(
                    "📥 Download Gold SOAP JSON",
                    data=json.dumps(gold_data, ensure_ascii=False, indent=2),
                    file_name=f"{browse_selected}.soap.json",
                    mime="application/json",
                    key="dl_gold_soap",
                )
        else:
            st.info(f"No gold SOAP file found for `{browse_selected}`. Generate one with the SOAP formatter.")

        # --- SOAP derived from Ground Truth (live) ---
        st.subheader("🔄 SOAP from Ground Truth (auto-derived)")
        with st.expander("View auto-derived SOAP sections from ground truth"):
            derived_sections = build_soap_sections(gt_dict)
            derived_text = soap_text_from_sections(derived_sections)
            st.text_area("Auto-derived SOAP", value=derived_text, height=250, disabled=True, key="derived_soap")
            st.json(derived_sections)

        # --- Past Evaluation Outputs ---
        st.subheader("📂 Past Evaluation Outputs")

        # Scan results directory for files matching this test case
        past_outputs = []
        if RESULTS_DIR.exists():
            # Medical facts results
            for p in sorted(RESULTS_DIR.glob("medical_facts_production_*.json")):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for entry in data:
                            if entry.get("test_name", "") == browse_tc.test_id or browse_selected in str(entry.get("test_name", "")):
                                past_outputs.append({"file": p.name, "type": "medical_facts", "data": entry})
                    elif isinstance(data, dict):
                        if data.get("test_name", "") == browse_tc.test_id or browse_selected in str(data.get("test_name", "")):
                            past_outputs.append({"file": p.name, "type": "medical_facts", "data": data})
                except Exception:
                    pass

            # SOAP results
            for p in sorted(RESULTS_DIR.glob("soap_evaluation_*.json")):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    results_list = data.get("results", []) if isinstance(data, dict) else []
                    for entry in results_list:
                        if entry.get("test_id", "") == browse_tc.test_id or browse_selected in str(entry.get("test_name", "")):
                            past_outputs.append({"file": p.name, "type": "soap", "data": entry})
                except Exception:
                    pass

            # SOAP agent per-test results
            soap_agents_dir = RESULTS_DIR / "soap_agents"
            if soap_agents_dir.exists():
                for eval_json in sorted(soap_agents_dir.rglob(f"{browse_selected}.soap_eval.json")):
                    try:
                        data = json.loads(eval_json.read_text(encoding="utf-8"))
                        past_outputs.append({
                            "file": str(eval_json.relative_to(RESULTS_DIR)),
                            "type": "soap_per_test",
                            "data": data,
                        })
                    except Exception:
                        pass

        if not past_outputs:
            st.info(f"No past evaluation outputs found for `{browse_selected}`.")
        else:
            st.caption(f"Found {len(past_outputs)} past evaluation(s) for this test case.")
            for i, po in enumerate(past_outputs):
                icon = "💊" if po["type"] == "medical_facts" else "🧼"
                label = f"{icon} {po['type']} — {po['file']}"
                agent_hint = po["data"].get("agent_id", "")[:12]
                score_hint = po["data"].get("quality_score") or po["data"].get("overall_score")
                if agent_hint:
                    label += f" — agent: {agent_hint}…"
                if score_hint is not None:
                    label += f" — score: {score_hint:.1f}"

                with st.expander(label):
                    # Show key metrics
                    if po["type"] == "medical_facts":
                        d = po["data"]
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Quality Score", f"{d.get('quality_score', 0):.1f}")
                        mc2.metric("Faithfulness", f"{d.get('faithfulness', 0):.1%}" if d.get("faithfulness") else "N/A")
                        mc3.metric("API Time", f"{d.get('api_time_seconds', 0):.2f}s")
                        if d.get("agent_output_raw"):
                            st.markdown("**Agent Output:**")
                            st.code(d["agent_output_raw"][:5000], language="json")
                    else:
                        d = po["data"]
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Overall", f"{d.get('overall_score', 0):.1f}/100")
                        mc2.metric("Structure", f"{d.get('structure_score', 0):.0%}")
                        mc3.metric("Content", f"{d.get('average_content_score', 0):.1%}")
                        mc4.metric("API Time", f"{d.get('api_time_seconds', 0):.2f}s")

                        # Show SOAP sections side-by-side
                        soap_pred = d.get("soap_sections", {})
                        soap_gold = d.get("gold_sections", {})
                        if soap_pred or soap_gold:
                            for sec in ("S", "O", "A", "P"):
                                pred_items = soap_pred.get(sec, [])
                                gold_items = soap_gold.get(sec, [])
                                st.markdown(f"**Section {sec}** — {len(pred_items)} predicted / {len(gold_items)} gold")
                                pc, gc = st.columns(2)
                                with pc:
                                    for item in pred_items:
                                        st.markdown(f"- {item}")
                                with gc:
                                    for item in gold_items:
                                        st.markdown(f"- _{item}_")

                        if d.get("raw_output"):
                            st.markdown("**Raw Agent Output:**")
                            st.code(d["raw_output"][:5000], language="json")

                    # Download individual result
                    st.download_button(
                        "📥 Download this result",
                        data=json.dumps(po["data"], ensure_ascii=False, indent=2),
                        file_name=f"{browse_selected}_{po['type']}_{i}.json",
                        mime="application/json",
                        key=f"dl_past_{i}",
                    )


# ========================== GOLD STANDARD REVIEW TAB ==========================
with tab_goldreview:
    st.header("📖 Gold Standard Review")
    st.caption(
        "Review all test cases: transcripts, gold SOAP notes, and ground truth medical facts. "
        "Leave feedback on what should be improved in each."
    )

    # --- Summary stats ---
    all_tc_names = list(test_cases.keys())
    gold_reviews_all = load_all_gold_reviews()
    reviewed_ids = {e.get("test_case_id") for e in gold_reviews_all}
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Test Cases", len(all_tc_names))
    sc2.metric("Gold SOAP Files", len(list(GOLD_SOAP_DIR.glob("*.soap.json"))))
    sc3.metric("Reviewed", f"{len(reviewed_ids & set(all_tc_names))}/{len(all_tc_names)}")

    # --- Filter / search ---
    gr_filter = st.text_input("🔍 Filter test cases", placeholder="Type to filter…", key="gr_filter")
    show_only_unreviewed = st.checkbox("Show only unreviewed", value=False, key="gr_unreviewed")

    filtered_tc_names = all_tc_names
    if gr_filter:
        filtered_tc_names = [n for n in filtered_tc_names if gr_filter.lower() in n.lower()]
    if show_only_unreviewed:
        filtered_tc_names = [n for n in filtered_tc_names if n not in reviewed_ids]

    st.markdown(f"Showing **{len(filtered_tc_names)}** of {len(all_tc_names)} test cases")
    st.markdown("---")

    # --- Per test case ---
    for tc_name in filtered_tc_names:
        tc_path = test_cases[tc_name]
        tc = load_test_case(tc_path)
        has_review = tc_name in reviewed_ids
        badge = "✅" if has_review else "⬜"
        gold_json_file = GOLD_SOAP_DIR / f"{tc_name}.soap.json"
        has_gold = gold_json_file.exists()

        with st.expander(f"{badge} **{tc_name}** {'🧼' if has_gold else '⚠️ no gold SOAP'}", expanded=False):

            # ---- Transcript ----
            st.markdown("### 🎙️ Transcript")
            st.markdown(
                f"<div style='height:200px; min-height:100px; max-height:80vh; overflow-y:auto; "
                f"background:#f8f9fa; padding:12px; border-radius:6px; border:1px solid #ddd; "
                f"font-size:14px; line-height:1.6; white-space:pre-wrap; user-select:text; "
                f"resize:vertical;'>"
                f"{tc.transcript}</div>",
                unsafe_allow_html=True,
            )

            col_left, col_right = st.columns(2)

            # ---- Ground Truth (Medical Facts) ----
            with col_left:
                st.markdown("### 📝 Ground Truth Medical Facts")
                gt = tc.ground_truth
                gt_dict = gt.to_dict()

                if gt.medications_taken:
                    st.markdown("**Medications Taken:**")
                    for m in gt.medications_taken:
                        parts = [f"**{m.name}**"]
                        if m.dose:
                            parts.append(m.dose)
                        if m.frequency:
                            parts.append(m.frequency)
                        if m.action:
                            parts.append(f"({m.action})")
                        st.markdown("- " + " — ".join(parts))

                if gt.medications_planned:
                    st.markdown("**Medications Planned:**")
                    for m in gt.medications_planned:
                        parts = [f"**{m.name}**"]
                        if m.dose:
                            parts.append(m.dose)
                        if m.frequency:
                            parts.append(m.frequency)
                        if m.reason:
                            parts.append(f"— {m.reason}")
                        st.markdown("- " + " — ".join(parts))

                if gt.symptoms:
                    st.markdown(f"**Symptoms:** {', '.join(gt.symptoms)}")
                if gt.medical_history:
                    st.markdown(f"**Medical History:** {', '.join(gt.medical_history)}")
                if gt.vital_measurements:
                    st.markdown("**Vitals:**")
                    for v in gt.vital_measurements:
                        st.markdown(f"- {v.parameter}: {v.value} {v.unit}")
                if gt.diagnostic_plans:
                    st.markdown(f"**Diagnostic Plans:** {', '.join(gt.diagnostic_plans)}")
                if gt.therapeutic_interventions:
                    st.markdown(f"**Therapeutic Interventions:** {', '.join(gt.therapeutic_interventions)}")
                if gt.forbidden_medications:
                    st.error(f"**Forbidden:** {', '.join(gt.forbidden_medications)}")

                with st.popover("View raw JSON"):
                    st.json(gt_dict)

            # ---- Gold SOAP ----
            with col_right:
                st.markdown("### 🧼 Gold Standard SOAP")
                if has_gold:
                    gold_data = json.loads(gold_json_file.read_text(encoding="utf-8"))
                    soap_sections = gold_data.get("soap", {})
                    for sec_label, sec_key in [("S — Subjective", "S"), ("O — Objective", "O"), ("A — Assessment", "A"), ("P — Plan", "P")]:
                        items = soap_sections.get(sec_key, [])
                        st.markdown(f"**{sec_label}** ({len(items)} items)")
                        for item in items:
                            st.markdown(f"- {item}")
                        if not items:
                            st.caption("(empty)")

                    with st.popover("View raw JSON"):
                        st.json(gold_data)
                else:
                    st.warning("No gold SOAP file found for this test case.")

            # ---- Existing feedback ----
            existing_reviews = load_gold_review(tc_name)
            if existing_reviews:
                st.markdown("### 📝 Previous Review Feedback")
                for rev in existing_reviews:
                    ts = rev.get("timestamp", "")[:16]
                    target = rev.get("target", "general")
                    st.markdown(f"> **{ts}** [{target}] — {rev.get('feedback', '')}")

            # ---- Feedback form ----
            st.markdown("### 💬 Leave Review Feedback")
            fb_target = st.selectbox(
                "What does this feedback apply to?",
                ["transcript", "ground_truth_medical_facts", "gold_soap", "general"],
                key=f"gr_tgt_{tc_name}",
            )
            fb_text = st.text_area(
                "Your feedback / improvement suggestions",
                placeholder="e.g. Missing vitals in gold SOAP section O, transcript has unclear medication dosage…",
                key=f"gr_fb_{tc_name}",
            )
            fb_priority = st.select_slider(
                "Priority",
                options=["low", "medium", "high", "critical"],
                value="medium",
                key=f"gr_pri_{tc_name}",
            )
            if st.button("💾 Save Review", key=f"gr_save_{tc_name}"):
                if fb_text.strip():
                    save_gold_review_entry(tc_name, {
                        "timestamp": datetime.now().isoformat(),
                        "test_case_id": tc_name,
                        "target": fb_target,
                        "priority": fb_priority,
                        "feedback": fb_text.strip(),
                    })
                    st.success("Review feedback saved!")
                    st.rerun()
                else:
                    st.warning("Enter some feedback text first.")

    # --- Export & manage ---
    st.markdown("---")
    ex1, ex2 = st.columns(2)
    with ex1:
        all_reviews = load_all_gold_reviews()
        if all_reviews:
            st.download_button(
                "📥 Export All Review Feedback",
                data=json.dumps(all_reviews, ensure_ascii=False, indent=2),
                file_name=f"gold_review_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="dl_gold_reviews",
            )
    with ex2:
        if all_reviews and st.button("🗑️ Clear All Review Feedback", type="secondary"):
            import shutil
            if GOLD_REVIEW_DIR.exists():
                shutil.rmtree(GOLD_REVIEW_DIR)
                GOLD_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
            st.success("All gold standard review feedback cleared.")
            st.rerun()


# ========================== RANKINGS TAB ==========================
with tab_ranking:
    st.header("🏆 Agent Rankings")

    rankings = load_rankings()
    if not rankings:
        st.info("No evaluation results yet. Run an evaluation to populate rankings.")
    else:
        # Aggregate by agent + mode
        from collections import defaultdict
        agg: dict[tuple, list] = defaultdict(list)
        for r in rankings:
            key = (r["agent_id"], r["mode"])
            agg[key].append(r)

        leaderboard = []
        for (aid, m), entries in agg.items():
            scores = [e["quality_score"] for e in entries if e.get("quality_score") is not None]
            times = [e["api_time_seconds"] for e in entries if e.get("api_time_seconds") is not None]
            passed = sum(1 for e in entries if e.get("passed"))
            failed = len(entries) - passed
            leaderboard.append({
                "Agent ID": f"{aid[:12]}…",
                "Full ID": aid,
                "Mode": m,
                "Runs": len(entries),
                "Avg Score": round(sum(scores) / len(scores), 1) if scores else 0,
                "Best Score": round(max(scores), 1) if scores else 0,
                "Worst Score": round(min(scores), 1) if scores else 0,
                "Avg Time (s)": round(sum(times) / len(times), 2) if times else 0,
                "Min Time (s)": round(min(times), 2) if times else 0,
                "Passed": passed,
                "Failed": failed,
                "Pass Rate": f"{(passed / len(entries) * 100):.0f}%" if entries else "0%",
                "Last Run": max(e["timestamp"] for e in entries),
            })

        # Sort by best score descending
        leaderboard.sort(key=lambda x: x["Best Score"], reverse=True)

        # Filter
        mode_filter = st.selectbox("Filter by mode", ["All", "medical_facts", "soap"])
        if mode_filter != "All":
            leaderboard = [r for r in leaderboard if r["Mode"] == mode_filter]

        # Display columns (hide Full ID in the table)
        display_cols = ["Agent ID", "Mode", "Runs", "Avg Score", "Best Score", "Worst Score", "Avg Time (s)", "Min Time (s)", "Pass Rate", "Last Run"]
        st.dataframe(
            [{k: r[k] for k in display_cols} for r in leaderboard],
            use_container_width=True,
            hide_index=True,
        )

        # Detail view per run
        st.subheader("📋 All Individual Runs")
        sort_by = st.selectbox("Sort by", ["timestamp", "quality_score", "api_time_seconds"], index=0)
        reverse = sort_by in ("quality_score",)
        sorted_rankings = sorted(rankings, key=lambda x: x.get(sort_by, ""), reverse=reverse)

        run_display = []
        for r in sorted_rankings[-100:]:  # last 100
            run_display.append({
                "Time": r.get("timestamp", "")[:19],
                "Agent": f"{r['agent_id'][:12]}…",
                "Mode": r["mode"],
                "Test Case": r.get("test_case", ""),
                "Score": round(r.get("quality_score", 0), 1),
                "Passed": "✅" if r.get("passed") else "❌",
                "API Time": f"{r.get('api_time_seconds', 0):.2f}s",
            })
        st.dataframe(run_display, use_container_width=True, hide_index=True)

        # Clear rankings
        if st.button("🗑️ Clear All Rankings", type="secondary"):
            save_rankings([])
            st.success("Rankings cleared.")
            st.rerun()

        # Export
        st.download_button(
            "📥 Download Rankings JSON",
            data=json.dumps(rankings, ensure_ascii=False, indent=2),
            file_name=f"rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# ========================== FEEDBACK TAB ==========================
with tab_feedback:
    st.header("💬 Feedback Browser")
    st.caption("View, filter, and export all feedback entries across test cases and agents.")

    all_fb = load_all_feedback()
    if not all_fb:
        st.info("No feedback yet. Run an evaluation and leave feedback using the 💬 form in results.")
    else:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        fb_agents = sorted({e["agent_id"] for e in all_fb})
        fb_modes = sorted({e.get("mode", "unknown") for e in all_fb})
        fb_tests = sorted({e.get("test_case_id", "unknown") for e in all_fb})
        with fc1:
            fb_agent_filter = st.selectbox("Filter by Agent", ["All"] + fb_agents, key="fb_agent_filter")
        with fc2:
            fb_mode_filter = st.selectbox("Filter by Mode", ["All"] + fb_modes, key="fb_mode_filter")
        with fc3:
            fb_test_filter = st.selectbox("Filter by Test Case", ["All"] + fb_tests, key="fb_test_filter")

        filtered = all_fb
        if fb_agent_filter != "All":
            filtered = [e for e in filtered if e["agent_id"] == fb_agent_filter]
        if fb_mode_filter != "All":
            filtered = [e for e in filtered if e.get("mode") == fb_mode_filter]
        if fb_test_filter != "All":
            filtered = [e for e in filtered if e.get("test_case_id") == fb_test_filter]

        st.markdown(f"**{len(filtered)}** feedback entries")

        for i, fb in enumerate(sorted(filtered, key=lambda x: x.get("timestamp", ""), reverse=True)):
            ts = fb.get("timestamp", "")[:16]
            aid_short = fb.get("agent_id", "?")[:12]
            tc_name = fb.get("test_case_name", fb.get("test_case_id", "?"))
            score = fb.get("score")
            score_str = f" — Score: {score:.1f}" if score is not None else ""
            header = f"{ts} | {fb.get('mode', '?')} | Agent {aid_short}… | {tc_name}{score_str}"
            with st.expander(header, expanded=False):
                st.markdown(f"**Feedback:** {fb.get('feedback', '—')}")
                recs = fb.get("recommendations", [])
                if recs:
                    st.markdown("**Auto-Recommendations:**")
                    for r in recs:
                        st.markdown(f"- {r}")
                st.caption(f"Agent: `{fb.get('agent_id', '?')}` | Test: `{fb.get('test_case_id', '?')}` | Full timestamp: {fb.get('timestamp', '?')}")

        # Export
        st.download_button(
            "📥 Download All Feedback JSON",
            data=json.dumps(filtered, ensure_ascii=False, indent=2),
            file_name=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

        # Clear all
        if st.button("🗑️ Clear All Feedback", type="secondary"):
            import shutil
            if FEEDBACK_DIR.exists():
                shutil.rmtree(FEEDBACK_DIR)
                FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
            st.success("All feedback cleared.")
            st.rerun()


# ========================== FAVORITES TAB ==========================
with tab_favorites:
    st.header("⭐ Favorite Agents")

    favorites = load_favorites()
    if not favorites:
        st.info("No favorites saved. Add one via the sidebar.")
    else:
        for i, f in enumerate(favorites):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                col1.markdown(f"**{f['label']}**")
                col2.code(f["agent_id"], language=None)
                col3.write(f"Mode: {f['mode']} | Added: {f.get('added', 'N/A')[:10]}")
                if col4.button("Delete", key=f"fav_del_{i}"):
                    favorites.pop(i)
                    save_favorites(favorites)
                    st.rerun()
            st.divider()

    st.subheader("Quick Add")
    with st.form("add_fav_form"):
        new_label = st.text_input("Label")
        new_id = st.text_input("Agent ID")
        new_mode = st.selectbox("Mode", ["medical_facts", "soap"])
        if st.form_submit_button("Add Favorite"):
            if new_label and new_id:
                favorites.append({
                    "label": new_label,
                    "agent_id": new_id,
                    "mode": new_mode,
                    "added": datetime.now().isoformat(),
                })
                save_favorites(favorites)
                st.success(f"Added: {new_label}")
                st.rerun()
