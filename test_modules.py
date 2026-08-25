from src.resume_parser import extract_resume_text
from src.skill_extractor import load_skills, extract_skills
from src.predictor import predict_top_roles


file_path = "test_resumes/nagarathna_cv_.pdf"

resume_text = extract_resume_text(file_path)

skill_list = load_skills()

detected_skills = extract_skills(
    resume_text,
    skill_list
)

top_roles = predict_top_roles(
    resume_text,
    detected_skills,
    top_n=3
)


print("\n" + "=" * 50)
print("        TOP JOB ROLE MATCHES")
print("=" * 50)

for i, item in enumerate(top_roles, start=1):

    print(
        f"{i}. {item['role']} "
        f"— {item['confidence']}%"
    )

print("=" * 50)