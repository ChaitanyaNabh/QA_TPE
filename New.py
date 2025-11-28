import streamlit as st
import PyPDF2
import io
import os
import json
from openai import OpenAI
from typing import List
from pdf_context import find_context_matches
from auth import verify_user, create_user, list_users


# OpenAI setup: expects OPENAI_API_KEY in env vars

# Embedded Medicare Benefit Policy reference (Chapter 7) to avoid re-uploading a manual PDF
POLICY_JSON = r'''{
  "schema_version": "1.0",
  "source_manual": "Medicare Benefit Policy Manual - Chapter 7 (Home Health Services)",
  "purpose": "Help a system check an OASIS/home-health case for Medicare coverage consistency with Chapter 7.",
  "notes": [
    "This JSON is a POLICY KNOWLEDGE BASE, not sample training dialogs.",
    "You can load this JSON and write code or prompts that ask GPT to apply these rules to OASIS data.",
    "Fill in the oasis_items with the fields from the specific OASIS version your agency uses (E, D1, etc.)."
  ],
  "oasis_use_cases": [
    "coverage_determination",
    "homebound_status",
    "skilled_need",
    "plan_of_care_requirements",
    "certification_and_face_to_face",
    "PDGM_case_mix_adjustment"
  ],
  "rules": [
    {
      "id": "eligibility.core_requirements",
      "title": "Core Medicare Home Health Eligibility Criteria",
      "manual_sections": ["Sec 20", "Sec 30"],
      "description": "Defines the baseline conditions a beneficiary must meet to qualify for Medicare home health coverage.",
      "oasis_items": [
        "<homebound-related items>",
        "<physician/allowed practitioner info>",
        "<skilled-nursing / therapy need items>",
        "<plan of care / services ordered>",
        "<certification/recertification tracking>"
      ],
      "logic": {
        "all_of": [
          "beneficiary_is_medicare_eligible",
          "hha_has_valid_medicare_agreement",
          "patient_meets_homebound_definition",
          "plan_of_care_established_and_reviewed_by_physician_or_allowed_practitioner",
          "patient_is_under_care_of_physician_or_allowed_practitioner",
          {
            "any_of": [
              "needs_intermittent_skilled_nursing",
              "needs_physical_therapy",
              "needs_speech_language_pathology",
              "has_continuing_need_for_occupational_therapy"
            ]
          },
          "services_claimed_are_covered_under_sections_40_50",
          "medicare_is_correct_payer",
          "services_not_otherwise_excluded"
        ]
      },
      "decision_labels": {
        "pass": "eligible_for_medicare_home_health_benefit",
        "fail": "not_eligible_for_medicare_home_health_benefit"
      }
    },
    {
      "id": "eligibility.homebound.criteria",
      "title": "Homebound (Confined to the Home) Criteria",
      "manual_sections": ["Sec 30.1.1"],
      "description": "Patient must meet BOTH Criterion One and Criterion Two to be considered homebound.",
      "oasis_items": [
        "<mobility/assistive-device items>",
        "<cognitive / behavioral limitations>",
        "<clinical narrative about ability to leave home>",
        "<recent outings / frequency and purpose>"
      ],
      "logic": {
        "all_of": [
          {
            "any_of": [
              "needs_supportive_device_or_assistance_to_leave_home",
              "requires_special_transport_to_leave_home",
              "leaving_home_is_medically_contraindicated"
            ]
          },
          {
            "all_of": [
              "normal_inability_to_leave_home",
              "leaving_home_requires_considerable_and_taxing_effort"
            ]
          }
        ]
      },
      "permitted_absences": {
        "do_not_break_homebound_status_if": [
          "absences_are_infrequent_or_of_short_duration",
          "absences_are_for_medical_treatment_eg_dialysis_chemo_radiation",
          "absences_are_for_licensed_or_certified_adult_day_care_medical_or_psychosocial_treatment",
          "occasional_nonmedical_events_eg_religious_service_barber_family_event_drive"
        ]
      },
      "decision_labels": {
        "pass": "homebound_status_met",
        "fail": "homebound_status_not_met"
      }
    },
    {
      "id": "eligibility.residence.definition",
      "title": "What Counts as Patient's Place of Residence",
      "manual_sections": ["Sec 30.1.2"],
      "description": "Defines what settings can be treated as the patient's 'home' for home health coverage.",
      "oasis_items": [
        "<patient_living_situation>",
        "<facility_type>",
        "<state_licensure_status_of_setting>"
      ],
      "logic": {
        "residence_is_home_if_any_of": [
          "patient_lives_in_private_dwelling_or_apartment",
          "patient_lives_in_relative_or_caregiver_home",
          "patient_lives_in_assisted_living_or_group_home_not_primarily_providing_inpatient_diagnostic_therapeutic_or_skilled_nursing_services"
        ],
        "residence_is_not_home_if_any_of": [
          "setting_meets_hospital_definition_under_Sec 1861(e)(1)",
          "setting_meets_SNF_definition_under_Sec 1819(a)(1)",
          "setting_is_most_medicaid_nursing_facilities_that_meet_SNF_or_hospital_standards"
        ],
        "assisted_living_duplicate_services_not_covered": "If services required under state licensure/base contract of the assisted living facility duplicate what the HHA would provide, those HHA services are not considered reasonable and necessary."
      },
      "decision_labels": {
        "pass": "setting_counts_as_home",
        "fail": "setting_does_not_count_as_home"
      }
    },
    {
      "id": "plan_of_care.requirements",
      "title": "Plan of Care Content and Specificity",
      "manual_sections": ["Sec 30.2", "Sec 30.2.2", "Sec 30.2.3"],
      "description": "Requirements for a valid plan of care and how orders must be written.",
      "oasis_items": [
        "<ordered_disciplines_and_frequencies>",
        "<measurable_goals>",
        "<visit_durations>",
        "<therapy_course_of_treatment>",
        "<verbal_order_tracking_if_used>"
      ],
      "logic": {
        "all_of": [
          "individualized_plan_of_care_is_present",
          "plan_of_care_is_based_on_comprehensive_assessment",
          "plan_lists_each_discipline_responsible_for_services",
          "plan_includes_frequency_and_duration_for_all_visits",
          "all_care_provided_matches_plan_of_care_orders"
        ],
        "if_therapy_included": {
          "all_of": [
            "course_of_therapy_established_by_physician_or_allowed_practitioner_after_consultation_with_therapist_if_needed",
            "plan_has_measurable_therapy_goals_related_to_illness_or_impairment",
            "expected_duration_of_therapy_is_documented",
            "course_of_treatment_is_consistent_with_therapist_assessment"
          ]
        },
        "order_specificity": {
          "frequency_must_be_specific_or_range": "Orders must specify visit frequency and duration; ranges are allowed but upper limit is treated as the ordered frequency (e.g. SN 2-4/wk x 4 wk).",
          "prn_orders": "PRN orders must describe the signs/symptoms that trigger a visit AND cap the maximum number of PRN visits before a new order is required."
        }
      },
      "decision_labels": {
        "pass": "plan_of_care_valid",
        "fail": "plan_of_care_incomplete_or_not_specific_enough"
      }
    },
    {
      "id": "certification.initial",
      "title": "Initial Certification Requirements (Start of Care)",
      "manual_sections": ["Sec 30.5.1", "Sec 30.5.1.1"],
      "description": "What a valid initial certification must state and how the face-to-face encounter requirement works.",
      "oasis_items": [
        "<start_of_care_date>",
        "<certifying_physician_or_allowed_practitioner>",
        "<face_to_face_date_and_provider>",
        "<homebound_statement>",
        "<skilled_need_statement>",
        "<plan_of_care_link>"
      ],
      "logic": {
        "certification_required_when": "A Start of Care OASIS is completed to initiate home health services.",
        "certification_must_attest_all_of": [
          "patient_is_homebound_per_manual_definition",
          "patient_needs_intermittent_skilled_nursing_or_PT_or_SLP_or_has_continuing_need_for_OT",
          "a_plan_of_care_has_been_established_and_will_be_periodICALLY_reviewed",
          "services_are_furnished_while_patient_is_under_care_of_physician_or_allowed_practitioner",
          "a_face_to_face_encounter_related_to_primary_reason_for_home_health_occurred_within_required_time_window"
        ],
        "face_to_face_requirements": {
          "time_window": "Encounter occurred no more than 90 days before or within 30 days after the start of home health care.",
          "allowed_performers": [
            "certifying_physician_or_allowed_practitioner",
            "physician_or_allowed_practitioner_with_privileges_who_cared_for_patient_in_acute_or_post_acute_setting_from_which_patient_was_directly_admitted",
            "allowed_NPPs_(NP_CNS_PA_certified_nurse_midwife)_working_in_required_collaboration_or_supervision_structure"
          ],
          "documentation": [
            "date_of_face_to_face_encounter",
            "that_the_encounter_related_to_primary_reason_for_home_health"
          ]
        },
        "timing_of_certification_completion": "Certification must be complete before the HHA bills Medicare, and should be completed when the plan of care is established or as soon as possible thereafter."
      },
      "decision_labels": {
        "pass": "certification_valid",
        "fail": "certification_missing_or_incomplete"
      }
    },
    {
      "id": "coverage.reasonable_and_necessary",
      "title": "Reasonable and Necessary Services Determination",
      "manual_sections": ["Sec 20.1.2", "Sec 20.2", "Sec 20.3"],
      "description": "How Medicare decides whether services documented in OASIS / plan of care are reasonable and necessary.",
      "oasis_items": [
        "<diagnoses_and_comorbidities>",
        "<functional_status_items>",
        "<wounds_pain_symptoms>",
        "<therapy_and_nursing_interventions>",
        "<caregiver_availability_and_willingness>",
        "<maintenance_program_if_any>"
      ],
      "logic": {
        "coverage_decision_must_be_based_on": [
          "individual_patient_plan_of_care",
          "OASIS_assessment_data_as_required_by_42_CFR_484.55",
          "clinical_record_for_the_specific_patient"
        ],
        "must_not_be_based_on": [
          "generic_diagnostic_screens_only",
          "numerical_utilization_screens_or_visit_count_norms",
          "assumptions_about_all_patients_with_same_diagnosis"
        ],
        "skilled_maintenance": "Skilled nursing or therapy is coverable for maintenance programs if skilled care is needed to maintain condition or prevent/slow further deterioration, regardless of improvement potential.",
        "caregiver_effect": {
          "rule": "If a family member or other person is actually providing services that fully meet the patient's needs, duplicate HHA services are not reasonable and necessary.",
          "presumption": "Assume no able and willing caregiver exists unless the patient/family says otherwise or the HHA has first-hand knowledge to the contrary.",
          "institutional_care": "Patient can still receive home health even if they would qualify for institutional care; supplemental non-skilled services do not affect coverage."
        }
      },
      "decision_labels": {
        "pass": "services_reasonable_and_necessary",
        "fail": "services_not_reasonable_or_necessary"
      }
    },
    {
      "id": "pdgm.case_mix",
      "title": "PDGM Case-Mix Variables Derived from OASIS and Claims",
      "manual_sections": ["Sec 10.2"],
      "description": "Describes the PDGM case-mix factors and where OASIS is used.",
      "oasis_items": [
        "<functional_impairment_items_used_for_PDGM>",
        "<admission_source_and_timing_if_captured>",
        "<clinical_group_determination_support>"
      ],
      "logic": {
        "case_mix_inputs_from_claims": [
          "admission_source_institutional_or_community",
          "timing_early_first_30_day_or_late_subsequent",
          "clinical_group_based_on_principal_diagnosis",
          "comorbidity_adjustment_based_on_secondary_diagnoses_(none_low_high)"
        ],
        "case_mix_input_from_oasis": [
          "functional_impairment_level_low_medium_or_high_based_on_specific_OASIS_items"
        ],
        "grouping": "Each 30-day period is grouped into one of 432 case-mix groups; each group has a relative weight representing expected cost."
      },
      "decision_labels": {
        "output": "pdgm_case_mix_profile"
      }
    }
  ]
}'''

# Interactive API key input if not set in environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

if not OPENAI_API_KEY:
    st.sidebar.markdown("<span class='brand-side'>HelpQA.ai</span>", unsafe_allow_html=True)
    OPENAI_API_KEY = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if not OPENAI_API_KEY:
    st.sidebar.warning("OpenAI API key required to use PDF LLM features.")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="OASIS TPE Checker", layout="wide")

# Initialize simple local auth session
if "user" not in st.session_state:
    st.session_state.user = None

# App styling for a cleaner/professional UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Sora:wght@500;600&display=swap');
    :root{
        --bg:#0c1220;
        --panel:#121a2c;
        --card-bg:#141f31;
        --card-border:#253447;
        --muted:#dbe1eb;
        --accent:#5a66f1;
        --accent-2:#45c6d6;
        --glow:0 14px 46px rgba(90,102,241,0.2);
    }
    body {background:#0c1220; font-family:'Space Grotesk', 'Sora', sans-serif; color:#f5f7fb;}
    .main .block-container {padding-top: 96px; padding-bottom: 64px; max-width: 1200px;}
    .top-nav {position:fixed; top:0; left:0; right:0; height:58px; background:linear-gradient(135deg,#0b1224,#111a33); border-bottom:1px solid #1f2937; display:flex; align-items:center; padding:0 28px; z-index:9999; box-shadow:0 12px 34px rgba(0,0,0,0.35); backdrop-filter: blur(8px);}
    .top-nav .brand {font-weight:900; color:#ffffff; letter-spacing:0.12em; text-transform:uppercase; font-size:1.2rem; text-shadow:0 6px 16px rgba(0,0,0,0.55);}
    .top-nav .brand-badge {background:#ffffff; color:#0b1224; padding:8px 12px; border-radius:12px; border:1px solid #22d3ee; box-shadow:0 8px 18px rgba(0,0,0,0.25);}
    body:has(section[data-testid="stSidebar"]) .top-nav {padding-left:260px;}
    .app-header {color:#e5e7eb;}
    .page-shell {max-width: 1200px; margin: 0 auto 18px; padding: 0 6px;}
    .hero {background: linear-gradient(135deg, #121a2c 0%, #0c1322 100%); border:1px solid rgba(255,255,255,0.06); border-radius:18px; padding:30px; box-shadow:0 16px 44px rgba(0,0,0,0.32); position:relative; overflow:hidden;}
    .hero:before {content:''; position:absolute; inset:-12% 18% 12% -12%; background: radial-gradient(circle at center, rgba(90,102,241,0.22), transparent 58%); filter: blur(52px);}
    .hero-grid {display:grid; grid-template-columns:1.2fr 0.8fr; gap:24px; align-items:center;}
    .eyebrow {text-transform:uppercase; letter-spacing:0.08em; font-weight:600; color:#e7ecff; font-size:0.9rem; margin-bottom:8px;}
    .hero h1 {margin:0 0 10px 0; color:#ffffff; font-size:2.3rem; text-shadow:0 4px 14px rgba(0,0,0,0.35);}
    .lead {color:#f0f3f8; max-width:720px; margin-bottom:16px; font-size:1.05rem;}
    .hero-tags {display:flex; flex-wrap:wrap; gap:10px; margin-top:8px;}
    .pill {background:linear-gradient(135deg, #5a66f1, #45c6d6); color:#0c1220; padding:9px 13px; border-radius:12px; font-weight:800; font-size:0.95rem; border:1px solid rgba(255,255,255,0.42); backdrop-filter: blur(8px);}
    .hero-cards {display:grid; gap:14px;}
    .stat-card {background:#141f31; border-radius:14px; padding:16px 18px; border:1px solid #2c3a50; box-shadow:var(--glow);}
    .stat-label {color:#e5e9f0; font-size:0.95rem; margin-bottom:6px;}
    .stat-value {font-size:1.36rem; font-weight:700; color:#f9fafb; margin-bottom:8px;}
    .stat-note {color:#f5f7fb; font-size:1rem;}
    .card {background: var(--card-bg); border:1px solid #2c3a50; border-radius:12px; padding:16px; box-shadow: 0 12px 34px rgba(0,0,0,0.26); margin-bottom:12px}
    .claim-header {font-weight:700; color:#f9fafb; margin-bottom:6px; font-size:1.08rem;}
    .summary {color: #f9fbfe; margin-bottom:14px; background:#18263b; padding:14px 16px; border-radius:12px; border:1px solid #2c3a50; box-shadow:0 10px 24px rgba(0,0,0,0.25);}
    .evidence {font-size:0.95rem; color:#f5f7fb; background:#1a2a40; padding:12px; border-radius:10px; margin-bottom:10px; border:1px solid #2c3a50;}
    .badge {display:inline-block; padding:7px 13px; border-radius:12px; font-weight:700; font-size:0.92rem}
    .badge-correct {background:#0f5132; color:#bbf7d0; border:1px solid #16a34a}
    .badge-incorrect {background:#5a1026; color:#fecdd3; border:1px solid #f43f5e}
    .badge-insufficient {background:#4a3500; color:#fef3c7; border:1px solid #f59e0b}
    .small-muted {color:#dfe5ef; font-size:0.9rem}
    /* Ensure readable body text */
    .main .block-container p,
    .main .block-container li,
    .main .block-container span {color:#f0f3f8;}
    div[data-testid="stCaptionContainer"] {color:#e2e7ee !important;}
    .sidebar-section {padding:8px 0 16px 0}
    .section-title {display:flex; align-items:center; gap:10px; margin:16px 0 6px; color:#f9fbfe; font-weight:800; text-shadow:0 2px 10px rgba(0,0,0,0.25);}
    .section-title span {height:12px; width:12px; border-radius:50%; background:linear-gradient(135deg, #5a66f1, #45c6d6); box-shadow:0 0 10px rgba(90,102,241,0.5);}
    .stTextArea textarea, .stTextInput input {border-radius:12px !important; border:1px solid #273246 !important; background:#0b1224 !important; color:#e5e7eb !important;}
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {color:#9ca3af !important;}
    section[data-testid="stSidebar"] {background:#0f1720;}
    section[data-testid="stSidebar"] .block-container {padding-top:24px; padding-bottom:24px;}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {color:#f8fafc;}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] label {color:#d8dee9;}
    section[data-testid="stSidebar"] .brand-side {font-weight:900; color:#ffffff; letter-spacing:0.14em; text-transform:uppercase; font-size:1.05rem; margin-bottom:14px; display:block;}
    section[data-testid="stSidebar"] .stButton button {width:100%; border-radius:10px; background:linear-gradient(135deg,#4f46e5,#22d3ee); color:#0b0c10; border:none; font-weight:700;}
    section[data-testid="stSidebar"] .stButton button:hover {filter:brightness(1.05);}
    section[data-testid="stSidebar"] .stFileUploader div {color:#d8dee9;}
    /* Sidebar toggle contrast */
    div[data-testid="collapsedControl"] button {background:#b91c1c !important; border:1px solid #7f1d1d !important; box-shadow:0 6px 16px rgba(185,28,28,0.35); border-radius:10px;}
    div[data-testid="collapsedControl"] button svg,
    div[data-testid="collapsedControl"] button svg path {color:#ffffff !important; fill:#ffffff !important; stroke:#ffffff !important;}
    div[data-testid="collapsedControl"] button[aria-label*="Show"],
    div[data-testid="collapsedControl"] button[aria-label*="Open"] {background:#fee2e2 !important; border:1px solid #b91c1c !important;}
    div[data-testid="collapsedControl"] button[aria-label*="Show"] svg,
    div[data-testid="collapsedControl"] button[aria-label*="Open"] svg,
    div[data-testid="collapsedControl"] button[aria-label*="Show"] svg path,
    div[data-testid="collapsedControl"] button[aria-label*="Open"] svg path {color:#7f1d1d !important; fill:#7f1d1d !important; stroke:#7f1d1d !important;}
    div[data-testid="stExpander"] > details {border:1px solid #1f2937; border-radius:12px; background:#0f172a; backdrop-filter: blur(6px); box-shadow:0 14px 42px rgba(0,0,0,0.28); margin-bottom:10px;}
    div[data-testid="stExpander"] summary {font-weight:700; color:#e5e7eb; font-size:1rem;}
    .result-grid {display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:12px; margin-top:10px;}
    .result-card {border-radius:12px; padding:12px; border:1px solid #1f2937; background:#0b1224;}
    .footer-note {color:#94a3b8; font-size:0.9rem; margin-top:12px;}
    button[kind="secondary"] {box-shadow:0 12px 26px rgba(79,70,229,0.25);}
    .stAlert > div {background:#0b1224 !important; border:1px solid #374151 !important; color:#e5e7eb !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

def _render_badge(verdict: str) -> str:
    v = (verdict or "").lower()
    if v == "correct":
        cls = "badge badge-correct"
        label = "Correct"
    elif v == "insufficient":
        cls = "badge badge-insufficient"
        label = "Insufficient"
    else:
        cls = "badge badge-incorrect"
        label = "Incorrect"
    return f"<span class='{cls}'>{label}</span>"

# Create a default user for convenience if it doesn't exist yet
try:
    create_user("Chaitanya", "Password@123")
except Exception:
    # ignore if user already exists or other creation error
    pass

# Sidebar: simple login/register UI
st.sidebar.header("Account")
if st.session_state.user:
    st.sidebar.write(f"Signed in as: **{st.session_state.user}**")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
else:
    # Prefill username for convenience
    login_username = st.sidebar.text_input("Username", value="Chaitanya", key="login_username")
    login_password = st.sidebar.text_input("Password", type="password", key="login_password")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("Login"):
            if verify_user(login_username, login_password):
                st.session_state.user = login_username
                st.sidebar.success("Logged in")
            else:
                st.sidebar.error("Invalid username or password")
    with c2:
        if st.button("Register"):
            try:
                create_user(login_username, login_password)
                st.session_state.user = login_username
                st.sidebar.success("Registered and logged in")
            except Exception as e:
                st.sidebar.error(str(e))

    # Developer convenience: reset default user password
    if st.sidebar.button("Reset default user password"):
        try:
            from auth import set_password
            set_password("Chaitanya", "Password@123")
            st.sidebar.success("Default user password reset to 'Password@123'")
        except Exception as e:
            st.sidebar.error(f"Failed to reset default user: {e}")


def extract_pdf_texts(uploaded_file) -> tuple:
    """Return (PdfReader, list of page texts) for uploaded PDF."""
    file_bytes = uploaded_file.read()
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    page_texts = []
    for i in range(len(reader.pages)):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            text = ""
        page_texts.append(text)
    return reader, page_texts


def build_pdf_info(reader, page_texts):
    return {"reader": reader, "pages": len(page_texts), "page_texts": page_texts}


def _build_pdf_report(case_result: dict) -> tuple:
    """Build a simple PDF from the case_result. Returns (bytes or None, error_or_none)."""
    try:
        from fpdf import FPDF  # lightweight PDF builder
    except Exception as e:
        return None, f"PDF export unavailable (fpdf not installed: {e})"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def _safe(text: str) -> str:
        # FPDF default fonts are latin-1; replace unsupported chars to avoid encode errors
        return (text or "").encode("latin-1", "replace").decode("latin-1")

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, _safe("OASIS vs Referral Audit"), ln=True)

    def add_heading(text):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, _safe(text), ln=True)
        pdf.set_font("Arial", "", 11)

    def add_text(text):
        pdf.multi_cell(0, 6, _safe(text))

    overall = case_result.get("overall_score")
    summary = case_result.get("summary", "")
    findings = case_result.get("findings", []) or []
    recs = case_result.get("recommended_actions", "")

    if overall is not None:
        add_heading(f"Overall Score: {overall}/100")
    if summary:
        add_heading("Summary")
        add_text(summary)

    if findings:
        add_heading("Findings")
        for idx, f in enumerate(findings, 1):
            status = f.get("status", "")
            claim = f.get("oasis_claim", "")
            issue = f.get("issue", "")
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 7, _safe(f"{idx}. [{status}] {claim}"), ln=True)
            pdf.set_font("Arial", "", 11)
            if issue:
                add_text(f"Issue: {issue}")
            if f.get("policy_refs"):
                add_text("Policy refs: " + ", ".join(f.get("policy_refs")))
            if f.get("citations"):
                add_text("Citations:")
                for c in f.get("citations"):
                    src = c.get("file", "")
                    page = c.get("page", "")
                    text = c.get("text", "")
                    add_text(f" - {src} p.{page}: {text}")
            if f.get("suggestion"):
                add_text(f"Suggestion: {f.get('suggestion')}")
            pdf.ln(2)

    if recs:
        add_heading("Recommended actions")
        add_text(recs if isinstance(recs, str) else json.dumps(recs, ensure_ascii=False))

    return pdf.output(dest="S").encode("latin-1", "replace"), None


def main():
    st.markdown(
        """
        <div class="top-nav"><div class="brand"><span class="brand-badge">HelpQA.ai</span></div></div>
        <div class="page-shell">
          <div class="hero">
            <div class="hero-grid">
              <div>
                <div class="eyebrow">OASIS QA workspace</div>
                <h1>OASIS Policy Auditor</h1>
                <p class="lead">Upload OASIS and referral PDFs, then auto-audit against Medicare Chapter 7 policy rules with citations, scores, and fix suggestions.</p>
                <div class="hero-tags">
                  <span class="pill">Policy-first</span>
                  <span class="pill">Cited evidence</span>
                  <span class="pill">Actionable fixes</span>
                </div>
              </div>
              <div class="hero-cards">
                <div class="stat-card">
                  <div class="stat-label">Step 1</div>
                  <div class="stat-value">Upload PDFs</div>
                  <div class="stat-note">Drop the OASIS file and referrals in the sidebar.</div>
                </div>
                <div class="stat-card">
                  <div class="stat-label">Step 2</div>
                  <div class="stat-value">Audit</div>
                  <div class="stat-note">Run the policy check to get findings, citations, and a score.</div>
                </div>
                <div class="stat-card">
                  <div class="stat-label">Step 3</div>
                  <div class="stat-value">Review</div>
                  <div class="stat-note">Read findings, export PDF, and chat for clarifications.</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Require login
    if not st.session_state.user:
        st.info("Please sign in using the Account box in the sidebar to use the app.")
        return

    # Checklist input moved to main area (not the sidebar)
    st.markdown("<div class='section-title'><span></span><h3>Case analysis</h3></div>", unsafe_allow_html=True)
    st.caption(":blue[Upload OASIS and referral/supporting PDFs. The app will apply Medicare Chapter 7 policy JSON to cross-check OASIS against referrals, cite sources, and suggest fixes.]")

    # Move file uploads to the sidebar
    st.sidebar.markdown("**Upload files**")
    oasis_file = st.sidebar.file_uploader("Upload OASIS PDF (required)", type="pdf", key="oasis_file")
    referral_files = st.sidebar.file_uploader(
        "Upload Referral / Supporting PDFs (one or more)", type="pdf", accept_multiple_files=True, key="referral_files"
    )

    if not oasis_file or not referral_files:
        st.info("Upload the OASIS PDF and at least one referral/supporting PDF to proceed.")
        return

    # Ensure OpenAI client available
    if client is None:
        st.warning("OpenAI API key not set - enter it in the sidebar to enable GPT queries.")
        return

    # Upload both PDFs to OpenAI Files API (if not already uploaded in session)
    if "openai_file_ids" not in st.session_state:
        st.session_state.openai_file_ids = {}

    # helper to upload and cache file_id
    def _upload_to_openai(key_name, file_obj):
        if key_name in st.session_state.openai_file_ids:
            return st.session_state.openai_file_ids[key_name]
        try:
            file_obj.seek(0)
            openai_file = client.files.create(file=file_obj, purpose="user_data")
            st.session_state.openai_file_ids[key_name] = openai_file.id
            return openai_file.id
        except Exception as e:
            st.error(f"Failed to upload {key_name} to OpenAI: {e}")
            return None

    with st.spinner("Uploading PDFs to OpenAI..."):
        file_id_oasis = _upload_to_openai("oasis", oasis_file)
        file_id_referrals = []
        for i, rf in enumerate(referral_files):
            fid = _upload_to_openai(f"referral_{i}", rf)
            if fid:
                file_id_referrals.append(fid)

    if not file_id_oasis or not file_id_referrals:
        st.error("One or more file uploads failed; cannot continue.")
        return

    st.success(f"Uploaded OASIS (file_id={file_id_oasis}) and {len(file_id_referrals)} referral file(s) to OpenAI")

    analyze_key = "analyze_case"
    if st.button("Analyze OASIS vs Referral", key=analyze_key):
        analysis_instruction = (
            "You are an auditor for OASIS/home health documentation. You have: (a) the OASIS PDF (claims), "
            "(b) one or more Referral/Supporting PDFs (source of truth), and (c) a policy JSON from Medicare Benefit Policy Manual Chapter 7. "
            "Task: cross-check the OASIS against referrals, applying the policy JSON. Identify incorrect or insufficient claims in OASIS, cite referral evidence (and policy sections), and suggest concrete fixes.\n\n"
            "Return EXACTLY one JSON object with keys:\n"
            "- overall_score: integer 0-100 reflecting compliance\n"
            "- summary: short overview (1-3 sentences)\n"
            "- findings: list of objects {status: 'Incorrect'|'Insufficient'|'Aligned', oasis_claim, issue, policy_refs: [strings], citations: [ {file: 'oasis'|'referral'|'policy', page: integer (policy use 0), text: string} ], suggestion: string}\n"
            "- recommended_actions: short list or string of prioritized remediation steps\n"
            "Keep citations concise; avoid raw PDF dumps."
        )

        with st.spinner("Analyzing case with policy JSON..."):
            try:
                file_inputs = [{"type": "input_file", "file_id": file_id_oasis}]
                for fid in file_id_referrals:
                    file_inputs.append({"type": "input_file", "file_id": fid})

                response = client.responses.create(
                    model=OPENAI_MODEL,
                    input=[
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": POLICY_JSON}] + file_inputs + [{"type": "input_text", "text": analysis_instruction}],
                        }
                    ],
                )
                out_text = response.output_text.strip()
                parsed = None
                try:
                    parsed = json.loads(out_text)
                except Exception:
                    s = out_text.find("{")
                    e = out_text.rfind("}")
                    if s != -1 and e != -1 and e > s:
                        try:
                            parsed = json.loads(out_text[s:e+1])
                        except Exception:
                            parsed = None

                if parsed and isinstance(parsed, dict):
                    st.session_state["case_result"] = parsed
                else:
                    st.error("Could not parse structured JSON from model response.")
            except Exception as e:
                st.error(f"Error from LLM: {e}")

    case_res = st.session_state.get("case_result")
    if case_res:
        overall_score = case_res.get("overall_score")
        summary = case_res.get("summary", "")
        findings = case_res.get("findings", [])
        recs = case_res.get("recommended_actions", "")

        st.markdown("### Findings")
        if overall_score is not None:
            st.markdown(f"**Overall score:** {overall_score}/100")
        if summary:
            st.markdown(f"<div class='summary'>{summary}</div>", unsafe_allow_html=True)

        status_badges = {
            "incorrect": "<span class='badge badge-incorrect'>Incorrect</span>",
            "insufficient": "<span class='badge badge-insufficient'>Insufficient</span>",
            "aligned": "<span class='badge badge-correct'>Aligned</span>",
        }

        if findings:
            for f in findings:
                status = (f.get("status") or "").lower()
                badge = status_badges.get(status, f"<span class='small-muted'>{status or 'Unknown'}</span>")
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"{badge} &nbsp; <strong>{f.get('oasis_claim','')}</strong>", unsafe_allow_html=True)
                if f.get("issue"):
                    st.write(f"Issue: {f.get('issue')}")
                if f.get("policy_refs"):
                    st.write(f"Policy refs: {', '.join(f.get('policy_refs'))}")
                if f.get("citations"):
                    st.markdown("**Citations:**")
                    for c in f.get("citations"):
                        src = c.get("file", "")
                        page = c.get("page", "")
                        text = c.get("text", "")
                        st.markdown(f"<div class='evidence'><strong>Source:</strong> {src} &nbsp; <span class='small-muted'>Page {page}</span><div>{text}</div></div>", unsafe_allow_html=True)
                if f.get("suggestion"):
                    st.write(f"Suggestion: {f.get('suggestion')}")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No findings returned.")

        if recs:
            st.markdown("**Recommended actions (overall):**")
            st.write(recs)

        # PDF export
        pdf_bytes, pdf_err = _build_pdf_report(case_res)
        if pdf_bytes:
            st.download_button(
                "Download audit as PDF",
                data=pdf_bytes,
                file_name="oasis_audit.pdf",
                mime="application/pdf",
            )
        elif pdf_err:
            st.info(pdf_err)

        # Case-level follow-up chat
        st.markdown("### Discuss / Ask follow-up")
        if "case_chat_history" not in st.session_state:
            st.session_state["case_chat_history"] = []

        with st.form(key="case_chat_form", clear_on_submit=True):
            user_q = st.text_area("Ask a follow-up question about this case", height=100)
            submitted = st.form_submit_button("Send")
            if submitted:
                if not user_q or user_q.strip() == "":
                    st.info("Enter a question to discuss this case.")
                elif client is None:
                    st.warning("OpenAI API key not set - enter it in the sidebar to ask GPT.")
                else:
                    follow_instruction = (
                        "You are a QA and medical coding assistant. Use the OASIS PDF, the referral/supporting PDFs, "
                        "and the Medicare Chapter 7 policy JSON provided. Answer the user's follow-up about this case. "
                        "Cite sources as {file: oasis|referral|policy, page: number (policy use 0), text: excerpt}. "
                        "Keep the answer concise and actionable.\n\n"
                        f"Overall score: {overall_score}\n"
                        f"Summary: {summary}\n"
                        f"User question: {user_q}\n"
                    )

                    file_inputs = [{"type": "input_file", "file_id": file_id_oasis}]
                    for fid in file_id_referrals:
                        file_inputs.append({"type": "input_file", "file_id": fid})

                    try:
                        response = client.responses.create(
                            model=OPENAI_MODEL,
                            input=[
                                {
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": POLICY_JSON}] + file_inputs + [{"type": "input_text", "text": follow_instruction}],
                                }
                            ],
                        )
                        reply = response.output_text.strip()
                    except Exception as e:
                        reply = f"Error from LLM: {e}"

                    st.session_state["case_chat_history"].append({"role": "user", "text": user_q})
                    st.session_state["case_chat_history"].append({"role": "assistant", "text": reply})

        if st.session_state.get("case_chat_history"):
            st.markdown("**Discussion history:**")
            for msg in st.session_state["case_chat_history"]:
                if msg.get("role") == "user":
                    st.markdown(f"**You:** {msg.get('text')}")
                else:
                    st.markdown(f"**Assistant:** {msg.get('text')}")
    else:
        st.info("Run 'Analyze OASIS vs Referral' to generate findings.")


if __name__ == "__main__":
    main()
