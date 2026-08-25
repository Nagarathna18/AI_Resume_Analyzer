import streamlit as st
import tempfile
import os

from src.analyzer import analyze_resume
from src.predictor import predict_top_roles


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* ---------- Header ---------- */

.main-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 6px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 25px;
    opacity: 0.75;
}


/* ---------- Section headings ---------- */

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 5px;
}


/* ---------- Upload area ---------- */

.upload-box {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128, 128, 128, 0.3);
    margin-top: 10px;
    margin-bottom: 15px;
}


/* ---------- Result cards ---------- */

.result-card {
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    margin-bottom: 15px;
}


/* ---------- Score ---------- */

.score-number {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
}


/* ---------- Skill badges ---------- */

.skill-badge {
    display: inline-block;
    padding: 7px 12px;
    margin: 4px;
    border-radius: 20px;
    border: 1px solid rgba(128, 128, 128, 0.3);
    font-size: 14px;
}


/* ---------- Footer ---------- */

.footer {
    text-align: center;
    margin-top: 40px;
    opacity: 0.6;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🤖 AI Resume Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered resume analysis, job role prediction & skill-gap analysis'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# INTRODUCTION
# ==================================================

st.write(
    "Upload your resume to analyze your skills, "
    "predict suitable job roles, identify skill gaps, "
    "and evaluate your resume against job requirements."
)


# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "📄 Upload your Resume",
    type=["pdf", "docx", "txt"]
)


# ==================================================
# ANALYZE
# ==================================================

if uploaded_file is not None:

    st.success(
        f"📎 {uploaded_file.name} uploaded successfully"
    )

    if st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True
    ):

        temp_path = None

        try:

            # ======================================
            # SAVE TEMPORARY FILE
            # ======================================

            file_extension = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name


            # ======================================
            # ANALYZE RESUME
            # ======================================

            with st.spinner(
                "🤖 Analyzing your resume..."
            ):

                result = analyze_resume(
                    temp_path
                )


            # ======================================
            # EXTRACT TEXT + SKILLS
            # ======================================

            from src.resume_parser import (
                extract_resume_text
            )

            from src.skill_extractor import (
                load_skills,
                extract_skills
            )

            resume_text = extract_resume_text(
                temp_path
            )

            skill_list = load_skills()

            detected_skills = extract_skills(
                resume_text,
                skill_list
            )


            # ======================================
            # TOP 3 ROLES
            # ======================================

            top_roles = predict_top_roles(
                resume_text,
                detected_skills,
                top_n=3
            )


            # ==================================================
            # TOP JOB ROLE MATCHES
            # ==================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '🎯 Top Job Role Matches'
                '</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "Ranking combines ML prediction with required-skill matching."
            )

            for i, role in enumerate(
                top_roles,
                start=1
            ):

                if i == 1:
                    rank_icon = "🥇"
                elif i == 2:
                    rank_icon = "🥈"
                else:
                    rank_icon = "🥉"

                with st.container(
                    border=True
                ):

                    st.subheader(
                        f"{rank_icon} {role['role']}"
                    )

                    st.write(
                        f"**Overall Match: "
                        f"{role['confidence']}%**"
                    )

                    st.progress(
                        min(
                            role["confidence"] / 100,
                            1.0
                        )
                    )

                    score_col1, score_col2 = st.columns(2)

                    with score_col1:

                        st.metric(
                            "🛠 Skill Match",
                            f"{role['skill_match']}%"
                        )

                    with score_col2:

                        st.metric(
                            "🤖 ML Score",
                            f"{role['ml_score']}%"
                        )


            # ==================================================
            # MAIN SUMMARY
            # ==================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '📋 Resume Summary'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)


            # --------------------------------------
            # PREDICTED ROLE
            # --------------------------------------

            with col1:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "🎯 Predicted Role"
                    )

                    st.success(
                        result["predicted_role"]
                    )


            # --------------------------------------
            # RESUME SCORE
            # --------------------------------------

            with col2:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "⭐ Resume Match Score"
                    )

                    st.metric(
                        "Overall Score",
                        f"{result['resume_score']}/100"
                    )

                    st.progress(
                        min(
                            result["resume_score"] / 100,
                            1.0
                        )
                    )


            # ==================================================
            # SKILLS SECTION
            # ==================================================

            st.divider()

            skill_col1, skill_col2 = st.columns(2)


            # --------------------------------------
            # DETECTED SKILLS
            # --------------------------------------

            with skill_col1:

                st.markdown(
                    '<div class="section-title">'
                    '🛠 Detected Skills'
                    '</div>',
                    unsafe_allow_html=True
                )

                if result["detected_skills"]:

                    skills_html = ""

                    for skill in result[
                        "detected_skills"
                    ]:

                        skills_html += (
                            f'<span class="skill-badge">'
                            f'✓ {skill}'
                            f'</span>'
                        )

                    st.markdown(
                        skills_html,
                        unsafe_allow_html=True
                    )

                else:

                    st.warning(
                        "No known skills were detected."
                    )


            # --------------------------------------
            # MISSING SKILLS
            # --------------------------------------

            with skill_col2:

                st.markdown(
                    '<div class="section-title">'
                    '❌ Skill Gaps'
                    '</div>',
                    unsafe_allow_html=True
                )

                if result["missing_skills"]:

                    for skill in result[
                        "missing_skills"
                    ]:

                        st.write(
                            f"❌ {skill}"
                        )

                else:

                    st.success(
                        "No missing required skills!"
                    )


            # ==================================================
            # SKILL MATCH
            # ==================================================

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '📊 Skill Match'
                '</div>',
                unsafe_allow_html=True
            )

            skill_percentage = result[
                "skill_match_percentage"
            ]

            st.progress(
                min(
                    skill_percentage / 100,
                    1.0
                )
            )

            st.metric(
                "Required Skills Matched",
                f"{skill_percentage}%"
            )


            # ==================================================
            # EDUCATION & EXPERIENCE
            # ==================================================

            st.divider()

            education_col, experience_col = st.columns(2)


            # --------------------------------------
            # EDUCATION
            # --------------------------------------

            with education_col:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "🎓 Education"
                    )

                    if result["education_meets"]:

                        st.success(
                            "✓ Meets requirement"
                        )

                    else:

                        st.warning(
                            "⚠ Requirement not detected"
                        )

                    if result[
                        "matched_education"
                    ]:

                        st.write(
                            "**Matched Education:**"
                        )

                        for education in result[
                            "matched_education"
                        ]:

                            st.write(
                                f"✓ {education}"
                            )


            # --------------------------------------
            # EXPERIENCE
            # --------------------------------------

            with experience_col:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "💼 Experience"
                    )

                    if result[
                        "candidate_experience"
                    ] is None:

                        st.write(
                            "Candidate experience:"
                        )

                        st.info(
                            "Not explicitly stated"
                        )

                        st.write(
                            f"Required: "
                            f"{result['required_experience']} "
                            f"years"
                        )

                        st.caption(
                            "Professional experience was "
                            "not explicitly stated in the resume."
                        )

                    else:

                        st.write(
                            f"Candidate: "
                            f"{result['candidate_experience']} "
                            f"years"
                        )

                        st.write(
                            f"Required: "
                            f"{result['required_experience']} "
                            f"years"
                        )

                        if result[
                            "experience_meets"
                        ]:

                            st.success(
                                "✓ Meets requirement"
                            )

                        else:

                            st.warning(
                                "⚠ Below requirement"
                            )


            # ==================================================
            # SALARY
            # ==================================================

            st.divider()

            with st.container(
                border=True
            ):

                st.subheader(
                    "💰 Salary Range"
                )

                st.info(
                    result["salary_range"]
                )


            # ==================================================
            # CATEGORY
            # ==================================================

            st.subheader(
                "🏷️ Job Category"
            )

            st.write(
                result["category"]
            )


            # ==================================================
            # FOOTER
            # ==================================================

            st.divider()

            st.markdown(
                '<div class="footer">'
                'AI Resume Analyzer • '
                'NLP + TF-IDF + Calibrated Linear SVM'
                '</div>',
                unsafe_allow_html=True
            )


        # ==================================================
        # ERROR HANDLING
        # ==================================================

        except Exception as e:

            st.error(
                "❌ Something went wrong while "
                "analyzing the resume."
            )

            st.exception(e)


        # ==================================================
        # CLEANUP
        # ==================================================

        finally:

            if temp_path is not None:

                try:

                    if os.path.exists(
                        temp_path
                    ):

                        os.remove(
                            temp_path
                        )

                except Exception:

                    pass