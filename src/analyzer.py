import re
import pandas as pd

from src.resume_parser import extract_resume_text
from src.skill_extractor import load_skills, extract_skills
from src.predictor import predict_job_role


# ==================================================
# LOAD DATA
# ==================================================

# Load job-role database
job_roles_df = pd.read_csv(
    "dataset/job_roles.csv"
)

# Load skill vocabulary
skill_list = load_skills()


# ==================================================
# EXPERIENCE EXTRACTION
# ==================================================

def extract_experience_years(resume_text):
    """
    Extract years of professional experience
    from the resume.

    Returns:
        int  -> if explicit experience is found
        None -> if experience is not explicitly stated
    """

    text = str(resume_text).lower()

    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:professional\s*)?(?:work\s*)?experience',
        r'(\d+)\+?\s*years?\s*(?:of\s*)?industry\s*experience',
        r'(\d+)\+?\s*years?\s*in\s*(?:the\s*)?(?:industry|field)'
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            return int(
                match.group(1)
            )

    # Experience was not explicitly stated
    return None


# ==================================================
# JOB ROLE REQUIREMENTS
# ==================================================

def get_role_requirements(predicted_role):
    """
    Get requirements for the predicted job role.
    """

    role_data = job_roles_df[
        job_roles_df["Job Title"].astype(str).str.lower()
        == predicted_role.lower()
    ]

    if role_data.empty:

        return None

    row = role_data.iloc[0]

    return {

        "education":
            str(
                row["Education Requirement"]
            ),

        "experience":
            int(
                row["Experience Years"]
            ),

        "salary":
            str(
                row["Salary Range"]
            ),

        "category":
            str(
                row["Category"]
            ),

        "required_skills":
            str(
                row["Required Skills"]
            )
    }


# ==================================================
# SKILL NORMALIZATION
# ==================================================

def normalize_skill_for_comparison(skill):
    """
    Convert equivalent skill names into a common
    representation for skill matching.
    """

    skill = str(
        skill
    ).lower().strip()

    aliases = {

        # ------------------------------------------
        # HTML / CSS
        # ------------------------------------------

        "html/css":
            "html_css",

        "html / css":
            "html_css",

        # ------------------------------------------
        # DATABASE
        # ------------------------------------------

        "database":
            "database",

        "databases":
            "database",

        "database management":
            "database",

        "database systems":
            "database",

        "mysql":
            "database",

        "postgresql":
            "database",

        "mongodb":
            "database",

        "oracle":
            "database",

        "sqlite":
            "database",

        # ------------------------------------------
        # REST API
        # ------------------------------------------

        "rest api":
            "rest_api",

        "rest apis":
            "rest_api",

        "restful api":
            "rest_api",

        "restful apis":
            "rest_api",

        # ------------------------------------------
        # NODE.JS
        # ------------------------------------------

        "node.js":
            "nodejs",

        "node js":
            "nodejs",

        "nodejs":
            "nodejs",

        # ------------------------------------------
        # SCRIPTING
        # ------------------------------------------

        "scripting":
            "scripting",

        "bash":
            "scripting",

        "shell":
            "scripting",

        "shell scripting":
            "scripting",

        "powershell":
            "scripting"
    }

    return aliases.get(
        skill,
        skill
    )


# ==================================================
# SKILL GAP ANALYSIS
# ==================================================

def skill_gap_analysis(
    predicted_role,
    detected_skills
):
    """
    Compare resume skills with required role skills.

    Equivalent skills are normalized before
    comparison.
    """

    role_data = get_role_requirements(
        predicted_role
    )

    if role_data is None:

        return [], [], []

    # ----------------------------------------------
    # Required skills
    # ----------------------------------------------

    required_skills = [

        skill.strip()

        for skill in role_data[
            "required_skills"
        ].split("|")

        if skill.strip()
    ]

    # ----------------------------------------------
    # Normalize detected skills
    # ----------------------------------------------

    detected_normalized = {

        normalize_skill_for_comparison(
            skill
        )

        for skill in detected_skills
    }

    # ----------------------------------------------
    # Scripting support
    #
    # The dataset uses "Scripting" as a broad
    # DevOps requirement.
    #
    # Python, Bash, Shell or PowerShell experience
    # can satisfy this requirement.
    # ----------------------------------------------

    has_python = any(
        str(skill).lower().strip()
        == "python"
        for skill in detected_skills
    )

    has_bash = any(
        str(skill).lower().strip()
        == "bash"
        for skill in detected_skills
    )

    has_shell = any(
        str(skill).lower().strip()
        in {
            "shell",
            "shell scripting"
        }
        for skill in detected_skills
    )

    has_powershell = any(
        str(skill).lower().strip()
        == "powershell"
        for skill in detected_skills
    )

    if (
        has_python
        or has_bash
        or has_shell
        or has_powershell
    ):

        detected_normalized.add(
            "scripting"
        )

    # ----------------------------------------------
    # Compare skills
    # ----------------------------------------------

    matched_skills = []
    missing_skills = []

    for skill in required_skills:

        required_normalized = (
            normalize_skill_for_comparison(
                skill
            )
        )

        # ------------------------------------------
        # HTML/CSS special case
        # ------------------------------------------

        if required_normalized == "html_css":

            has_html = any(

                normalize_skill_for_comparison(
                    detected_skill
                ) == "html"

                for detected_skill
                in detected_skills
            )

            has_css = any(

                normalize_skill_for_comparison(
                    detected_skill
                ) == "css"

                for detected_skill
                in detected_skills
            )

            if has_html and has_css:

                matched_skills.append(
                    skill
                )

            else:

                missing_skills.append(
                    skill
                )

        # ------------------------------------------
        # Normal skill comparison
        # ------------------------------------------

        elif (
            required_normalized
            in detected_normalized
        ):

            matched_skills.append(
                skill
            )

        else:

            missing_skills.append(
                skill
            )

    return (
        required_skills,
        matched_skills,
        missing_skills
    )


# ==================================================
# EDUCATION ANALYSIS
# ==================================================

def check_education(
    resume_text,
    education_requirement
):
    """
    Check whether the resume contains
    one of the accepted education requirements.
    """

    resume_lower = str(
        resume_text
    ).lower()

    education_options = [

        edu.strip()

        for edu in str(
            education_requirement
        ).split("|")

        if edu.strip()
    ]

    matched_education = []

    for education in education_options:

        education_words = re.findall(
            r'\b[a-zA-Z]+\b',
            education.lower()
        )

        important_words = [

            word

            for word in education_words

            if word not in {
                "in",
                "and",
                "of",
                "or"
            }
        ]

        if not important_words:

            continue

        matches = sum(

            word in resume_lower

            for word in important_words
        )

        # At least 60% of important words
        # should be present
        if matches >= max(
            1,
            len(important_words) * 0.6
        ):

            matched_education.append(
                education
            )

    return matched_education


# ==================================================
# RESUME MATCH SCORE
# ==================================================

def calculate_resume_score(
    skill_match_percentage,
    education_matched,
    candidate_experience,
    required_experience
):
    """
    Calculate transparent resume-role match score.

    Weight:
        Skills     = 60%
        Education  = 20%
        Experience = 20%
    """

    # ----------------------------------------------
    # Skills = 60%
    # ----------------------------------------------

    skill_score = (
        skill_match_percentage
        * 0.60
    )

    # ----------------------------------------------
    # Education = 20%
    # ----------------------------------------------

    if education_matched:

        education_score = 20

    else:

        education_score = 0

    # ----------------------------------------------
    # Experience = 20%
    # ----------------------------------------------

    if candidate_experience is None:

        # Experience was not stated.
        # Do not assume 0 years.
        experience_score = 0

    elif required_experience <= 0:

        experience_score = 20

    else:

        experience_ratio = min(

            candidate_experience
            / required_experience,

            1
        )

        experience_score = (
            experience_ratio
            * 20
        )

    # ----------------------------------------------
    # Total score
    # ----------------------------------------------

    total_score = (

        skill_score
        + education_score
        + experience_score
    )

    return round(
        total_score,
        2
    )


# ==================================================
# COMPLETE RESUME ANALYSIS
# ==================================================

def analyze_resume(file_path):
    """
    Perform complete resume analysis.
    """

    # ----------------------------------------------
    # 1. Extract resume text
    # ----------------------------------------------

    resume_text = extract_resume_text(
        file_path
    )

    # ----------------------------------------------
    # 2. Extract skills
    # ----------------------------------------------

    detected_skills = extract_skills(
        resume_text,
        skill_list
    )

    # ----------------------------------------------
    # 3. Predict job role
    # ----------------------------------------------

    predicted_role = predict_job_role(
        resume_text,
        detected_skills
    )

    # ----------------------------------------------
    # 4. Get role requirements
    # ----------------------------------------------

    role_requirements = get_role_requirements(
        predicted_role
    )

    # ----------------------------------------------
    # Safety check
    # ----------------------------------------------

    if role_requirements is None:

        return {

            "predicted_role":
                predicted_role,

            "detected_skills":
                detected_skills,

            "required_skills":
                [],

            "matched_skills":
                [],

            "missing_skills":
                [],

            "skill_match_percentage":
                0,

            "candidate_experience":
                None,

            "required_experience":
                0,

            "experience_meets":
                False,

            "education_requirements":
                "",

            "matched_education":
                [],

            "education_meets":
                False,

            "salary_range":
                "Not available",

            "category":
                "Unknown",

            "resume_score":
                0
        }

    # ----------------------------------------------
    # 5. Skill gap analysis
    # ----------------------------------------------

    (
        required_skills,
        matched_skills,
        missing_skills
    ) = skill_gap_analysis(
        predicted_role,
        detected_skills
    )

    # ----------------------------------------------
    # Calculate skill match
    # ----------------------------------------------

    if required_skills:

        skill_match = (

            len(matched_skills)
            / len(required_skills)

        ) * 100

    else:

        skill_match = 0

    # ----------------------------------------------
    # 6. Experience analysis
    # ----------------------------------------------

    candidate_experience = (
        extract_experience_years(
            resume_text
        )
    )

    required_experience = (
        role_requirements[
            "experience"
        ]
    )

    if candidate_experience is None:

        experience_meets = False

    else:

        experience_meets = (

            candidate_experience
            >= required_experience
        )

    # ----------------------------------------------
    # 7. Education analysis
    # ----------------------------------------------

    matched_education = check_education(
        resume_text,
        role_requirements[
            "education"
        ]
    )

    education_meets = (
        len(matched_education) > 0
    )

    # ----------------------------------------------
    # 8. Resume match score
    # ----------------------------------------------

    resume_score = calculate_resume_score(

        skill_match,

        education_meets,

        candidate_experience,

        required_experience
    )

    # ----------------------------------------------
    # 9. Return complete results
    # ----------------------------------------------

    return {

        "predicted_role":
            predicted_role,

        "detected_skills":
            detected_skills,

        "required_skills":
            required_skills,

        "matched_skills":
            matched_skills,

        "missing_skills":
            missing_skills,

        "skill_match_percentage":
            round(
                skill_match,
                2
            ),

        "candidate_experience":
            candidate_experience,

        "required_experience":
            required_experience,

        "experience_meets":
            experience_meets,

        "education_requirements":
            role_requirements[
                "education"
            ],

        "matched_education":
            matched_education,

        "education_meets":
            education_meets,

        "salary_range":
            role_requirements[
                "salary"
            ],

        "category":
            role_requirements[
                "category"
            ],

        "resume_score":
            resume_score
    }