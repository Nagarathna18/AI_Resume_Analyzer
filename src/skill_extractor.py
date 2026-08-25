import pandas as pd
import re


# ==================================================
# LOAD SKILLS
# ==================================================

def load_skills():
    """Load and clean the skill vocabulary."""

    skills_df = pd.read_csv(
        "dataset/skills_list.csv"
    )

    skill_list = (
        skills_df["Skill Name"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    # Skills that should always be recognized
    essential_skills = [
        # Programming
        "Python",
        "Java",
        "C++",
        "JavaScript",
        "HTML",
        "CSS",

        # Version Control
        "Git",
        "GitHub",

        # Software Development
        "Software Design",
        "System Design",
        "Data Structures",
        "Algorithms",

        # APIs
        "REST API",

        # Web
        "Django",
        "Flask",
        "Spring",
        "Spring Boot",
        "React",
        "Node.js",
        "Express",
        "HTML/CSS",
        "Responsive Design",

        # Databases
        "SQL",
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "Database",

        # Data / AI
        "Machine Learning",
        "Deep Learning",
        "Data Science",
        "Data Analysis",
        "Statistics",
        "Scikit-learn",
        "Pandas",
        "NumPy",
        "TensorFlow",
        "PyTorch",
        "Jupyter",

        # Cloud
        "AWS",
        "Azure",
        "GCP",

        # DevOps
        "Docker",
        "Kubernetes",
        "Terraform",
        "Jenkins",
        "CI/CD",

        # Systems
        "Linux",
        "Networking",
        "Security",

        # Soft Skills
        "Communication",
        "Problem Solving",
        "Teamwork",
        "Project Management",
        "Time Management",
        "Agile"
    ]

    # Existing skills in lowercase
    existing_skills = {
        skill.lower()
        for skill in skill_list
    }

    # Add essential skills if missing
    for skill in essential_skills:

        if skill.lower() not in existing_skills:

            skill_list.append(skill)

            existing_skills.add(
                skill.lower()
            )

    return skill_list


# ==================================================
# TEXT NORMALIZATION
# ==================================================

def normalize_text(text):
    """
    Normalize text for reliable skill matching.
    """

    text = str(text).lower()

    # Normalize Unicode dashes
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # Normalize slash spacing
    text = re.sub(
        r"\s*/\s*",
        "/",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==================================================
# SKILL ALIASES
# ==================================================

SKILL_ALIASES = {

    # ----------------------------------------------
    # Programming
    # ----------------------------------------------

    "c plus plus": "C++",
    "cpp": "C++",

    "js": "JavaScript",

    # ----------------------------------------------
    # Git
    # ----------------------------------------------

    "git hub": "GitHub",
    "version control": "Git",

    # ----------------------------------------------
    # APIs
    # ----------------------------------------------

    "restful api": "REST API",
    "restful apis": "REST API",
    "rest api": "REST API",
    "rest apis": "REST API",

    # ----------------------------------------------
    # Node.js
    # ----------------------------------------------

    "nodejs": "Node.js",
    "node js": "Node.js",

    # ----------------------------------------------
    # Machine Learning
    # ----------------------------------------------

    "ml": "Machine Learning",

    "deep neural networks": "Deep Learning",
    "neural networks": "Deep Learning",

    # ----------------------------------------------
    # SQL
    # ----------------------------------------------

    "structured query language": "SQL",

    # ----------------------------------------------
    # Cloud
    # ----------------------------------------------

    "google cloud": "GCP",
    "google cloud platform": "GCP",

    "amazon web services": "AWS",

    "microsoft azure": "Azure",

    # ----------------------------------------------
    # Software Design
    # ----------------------------------------------

    "software architecture": "Software Design",

    "software design": "Software Design",

    # ----------------------------------------------
    # HTML / CSS
    # ----------------------------------------------

    "html/css": "HTML/CSS",
    "html / css": "HTML/CSS",
    "html and css": "HTML/CSS",

    # ----------------------------------------------
    # Database
    # ----------------------------------------------

    "database management": "Database",
    "database systems": "Database",
    "databases": "Database",

    # ----------------------------------------------
    # Data Analysis
    # ----------------------------------------------

    "data analytics": "Data Analysis",

    # ----------------------------------------------
    # Scikit-learn
    # ----------------------------------------------

    "sklearn": "Scikit-learn",
    "scikit learn": "Scikit-learn"
}


# ==================================================
# EXTRACT SKILLS
# ==================================================

def extract_skills(text, skill_list):
    """
    Extract known skills from resume text.

    Uses:
        1. Exact phrase matching
        2. Skill aliases
        3. Canonical skill names
        4. Duplicate removal
    """

    normalized_text = normalize_text(
        text
    )

    found_skills = []


    # ==================================================
    # 1. MATCH SKILLS FROM VOCABULARY
    # ==================================================

    for skill in skill_list:

        skill_original = skill.strip()

        if not skill_original:
            continue

        skill_normalized = normalize_text(
            skill_original
        )

        pattern = (
            r"(?<!\w)"
            + re.escape(skill_normalized)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            normalized_text
        ):

            # ------------------------------------------
            # Canonicalize duplicate variants
            # ------------------------------------------

            if skill_normalized in {
                "rest apis",
                "restful api",
                "restful apis"
            }:

                canonical_skill = "REST API"

            elif skill_normalized in {
                "html",
                "css"
            }:

                canonical_skill = skill_original

            else:

                canonical_skill = skill_original

            if canonical_skill not in found_skills:

                found_skills.append(
                    canonical_skill
                )


    # ==================================================
    # 2. MATCH ALIASES
    # ==================================================

    for alias, canonical_skill in SKILL_ALIASES.items():

        alias_normalized = normalize_text(
            alias
        )

        pattern = (
            r"(?<!\w)"
            + re.escape(alias_normalized)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            normalized_text
        ):

            if canonical_skill not in found_skills:

                found_skills.append(
                    canonical_skill
                )


    # ==================================================
    # 3. HTML + CSS → HTML/CSS
    # ==================================================

    has_html = any(
        skill.lower() == "html"
        for skill in found_skills
    )

    has_css = any(
        skill.lower() == "css"
        for skill in found_skills
    )

    if has_html and has_css:

        if "HTML/CSS" not in found_skills:

            found_skills.append(
                "HTML/CSS"
            )


    # ==================================================
    # 4. DATABASE DETECTION
    # ==================================================

    database_skills = {
        "mysql",
        "postgresql",
        "mongodb",
        "oracle",
        "sqlite",
        "sql server"
    }

    has_database = any(
        skill.lower() in database_skills
        for skill in found_skills
    )

    if has_database:

        if "Database" not in found_skills:

            found_skills.append(
                "Database"
            )


    # ==================================================
    # 5. REMOVE DUPLICATES
    # ==================================================

    unique_skills = []

    seen = set()

    for skill in found_skills:

        key = skill.lower().strip()

        if key not in seen:

            seen.add(key)

            unique_skills.append(
                skill
            )


    return unique_skills