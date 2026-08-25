import joblib
import pandas as pd


MODEL_PATH = "models/calibrated_job_role_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
JOB_ROLES_PATH = "dataset/job_roles.csv"


def load_model():
    """Load the calibrated job-role model and TF-IDF vectorizer."""

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def load_job_roles():
    """Load job-role requirements from the job roles dataset."""

    return pd.read_csv(JOB_ROLES_PATH)


def prepare_text(resume_text, detected_skills):
    """Combine resume text and detected skills."""

    skills_text = " ".join(detected_skills)

    return str(resume_text) + " " + skills_text


def predict_job_role(resume_text, detected_skills):
    """Predict the most likely job role."""

    model, vectorizer = load_model()

    ml_text = prepare_text(
        resume_text,
        detected_skills
    )

    resume_tfidf = vectorizer.transform(
        [ml_text]
    )

    return model.predict(resume_tfidf)[0]


def calculate_skill_match(
    detected_skills,
    required_skills
):
    """
    Calculate percentage of required skills
    that are present in the resume.
    """

    detected = {
        str(skill).strip().lower()
        for skill in detected_skills
    }

    required = {
        str(skill).strip().lower()
        for skill in required_skills
        if str(skill).strip()
    }

    if not required:
        return 0.0

    matched = detected.intersection(required)

    return round(
        (len(matched) / len(required)) * 100,
        2
    )


def predict_top_roles(
    resume_text,
    detected_skills,
    top_n=3
):
    """
    Return top job-role matches using both:

    1. ML model confidence
    2. Required skill overlap

    Final score:
        40% ML score
        60% skill match
    """

    model, vectorizer = load_model()
    job_roles = load_job_roles()

    # --------------------------------------
    # Prepare resume text
    # --------------------------------------

    ml_text = prepare_text(
        resume_text,
        detected_skills
    )

    resume_tfidf = vectorizer.transform(
        [ml_text]
    )

    # --------------------------------------
    # Get ML probabilities
    # --------------------------------------

    probabilities = model.predict_proba(
        resume_tfidf
    )[0]

    classes = model.classes_

    max_probability = probabilities.max()

    results = []

    # --------------------------------------
    # Evaluate every job role
    # --------------------------------------

    for index, role in enumerate(classes):

        role = str(role)

        ml_probability = float(
            probabilities[index]
        )

        # ----------------------------------
        # Normalize ML score
        # ----------------------------------
        # The probabilities are spread across
        # many classes, so we compare each
        # probability with the highest one.

        if max_probability > 0:

            ml_score = (
                ml_probability /
                max_probability
            ) * 100

        else:

            ml_score = 0.0

        # ----------------------------------
        # Find role requirements
        # ----------------------------------

        role_data = job_roles[
            job_roles["Job Title"].astype(str).str.strip().str.lower()
            == role.strip().lower()
        ]

        skill_match = 0.0

        if not role_data.empty:

            required_skills_text = str(
                role_data.iloc[0]["Required Skills"]
            )

            required_skills = [
                skill.strip()
                for skill in required_skills_text.split("|")
                if skill.strip()
            ]

            skill_match = calculate_skill_match(
                detected_skills,
                required_skills
            )

        # ----------------------------------
        # Combined score
        # ----------------------------------

        final_score = (
            (ml_score * 0.40) +
            (skill_match * 0.60)
        )

        results.append({

            "role": role,

            "confidence": round(
                final_score,
                2
            ),

            "ml_score": round(
                ml_score,
                2
            ),

            "skill_match": round(
                skill_match,
                2
            )
        })

    # --------------------------------------
    # Sort by final match score
    # --------------------------------------

    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return results[:top_n]