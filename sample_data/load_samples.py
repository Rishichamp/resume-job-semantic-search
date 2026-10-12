"""
Sample data loader — populates the index with realistic test data.
Run: python sample_data/load_samples.py
"""
import requests, json, sys

BASE = "http://localhost:8000"

RESUMES = [
    {
        "resume_id": "r001",
        "skills": ["python", "pytorch", "transformers", "llms", "fastapi", "docker", "kubernetes", "mlflow"],
        "experience_summary": (
            "5 years in machine learning engineering. Built and deployed transformer-based NLP models "
            "at scale. Led a team of 3 engineers to build a real-time recommendation system processing "
            "2M events/day. Published research on attention mechanisms."
        ),
        "education": "M.Tech Artificial Intelligence, IIT (ISM) Dhanbad, 2023",
        "certifications": ["AWS Certified ML Specialist", "Google Professional ML Engineer"],
        "projects": "Lighthouse Attention (hierarchical attention for LLMs), MLOps pipeline with Kubernetes HPA",
        "years_experience": 5.0,
        "level": "senior",
        "candidate_name": "Arjun Sharma",
        "email": "arjun.sharma@email.com",
        "location": "Bangalore, India",
    },
    {
        "resume_id": "r002",
        "skills": ["python", "langchain", "rag", "openai api", "chromadb", "fastapi", "react", "sql"],
        "experience_summary": (
            "3 years building GenAI applications. Developed a RAG-based customer support system "
            "handling 10K queries/day. Built prompt evaluation frameworks. Strong in full-stack AI apps."
        ),
        "education": "B.Tech Computer Science, BITS Pilani, 2021",
        "certifications": ["DeepLearning.AI LangChain Developer"],
        "projects": "Enterprise RAG helpdesk, AI-powered code review tool",
        "years_experience": 3.0,
        "level": "mid",
        "candidate_name": "Priya Nair",
        "email": "priya.nair@email.com",
        "location": "Hyderabad, India",
    },
    {
        "resume_id": "r003",
        "skills": ["python", "scikit-learn", "pandas", "numpy", "sql", "tableau", "statistics", "a/b testing"],
        "experience_summary": (
            "2 years as a data scientist. Built predictive models for churn prediction (AUC 0.89). "
            "Created dashboards for business intelligence. Experience with feature engineering and model interpretability."
        ),
        "education": "B.Sc Statistics, University of Delhi, 2022",
        "certifications": [],
        "projects": "Customer churn prediction, Sales forecasting dashboard",
        "years_experience": 2.0,
        "level": "junior",
        "candidate_name": "Rahul Verma",
        "email": "rahul.verma@email.com",
        "location": "Delhi, India",
    },
    {
        "resume_id": "r004",
        "skills": ["python", "tensorflow", "keras", "computer vision", "opencv", "yolo", "docker", "gcp"],
        "experience_summary": (
            "7 years in deep learning, specialising in computer vision. Deployed production CV systems "
            "for manufacturing defect detection (99.2% precision). Strong in object detection and segmentation."
        ),
        "education": "PhD Computer Science, IIT Bombay, 2018",
        "certifications": ["Google Cloud Professional Data Engineer"],
        "projects": "Real-time defect detection pipeline, Medical image segmentation",
        "years_experience": 7.0,
        "level": "lead",
        "candidate_name": "Dr. Meera Krishnan",
        "email": "meera.k@email.com",
        "location": "Mumbai, India",
    },
    {
        "resume_id": "r005",
        "skills": ["python", "huggingface", "fine-tuning", "lora", "qlora", "peft", "nlp", "bert"],
        "experience_summary": (
            "1 year experience, fresh graduate specialising in LLM fine-tuning. "
            "Fine-tuned LLaMA 3 on domain-specific data using LoRA. Strong research background."
        ),
        "education": "B.Tech AI, VIT Vellore, 2024",
        "certifications": ["HuggingFace NLP Course"],
        "projects": "Domain-adapted LLaMA for legal text, BERT sentiment classifier",
        "years_experience": 1.0,
        "level": "junior",
        "candidate_name": "Sneha Patel",
        "email": "sneha.patel@email.com",
        "location": "Pune, India",
    },
]

JOBS = [
    {
        "job_id": "j001",
        "title": "Senior ML Engineer — LLMs & GenAI",
        "company": "TechCorp AI",
        "description": (
            "We are looking for a senior ML engineer to lead LLM deployment and optimisation. "
            "You will build and deploy transformer-based models, design MLOps pipelines, "
            "and mentor junior engineers. Experience with RAG systems is a plus."
        ),
        "required_skills": ["python", "pytorch", "transformers", "mlops", "docker", "kubernetes"],
        "nice_to_have_skills": ["triton", "rag", "langchain"],
        "level": "senior",
        "min_years_experience": 4.0,
        "location": "Bangalore, India",
        "remote": True,
        "salary_range": "₹30–50 LPA",
    },
    {
        "job_id": "j002",
        "title": "GenAI Application Developer",
        "company": "StartupXYZ",
        "description": (
            "Build production GenAI applications using LangChain, OpenAI, and vector databases. "
            "You will design RAG pipelines, integrate LLMs into web applications, "
            "and own the full stack from prompt engineering to deployment."
        ),
        "required_skills": ["python", "langchain", "openai api", "fastapi", "chromadb"],
        "nice_to_have_skills": ["react", "sql", "docker"],
        "level": "mid",
        "min_years_experience": 2.0,
        "location": "Hyderabad, India",
        "remote": False,
        "salary_range": "₹15–25 LPA",
    },
    {
        "job_id": "j003",
        "title": "Junior Data Scientist",
        "company": "AnalyticsCo",
        "description": (
            "Entry-level data science role. You will build predictive models, "
            "conduct statistical analysis, create dashboards, and support A/B testing. "
            "Strong Python and SQL skills required."
        ),
        "required_skills": ["python", "scikit-learn", "sql", "pandas", "statistics"],
        "nice_to_have_skills": ["tableau", "spark"],
        "level": "junior",
        "min_years_experience": 0.0,
        "location": "Delhi, India",
        "remote": True,
        "salary_range": "₹6–10 LPA",
    },
    {
        "job_id": "j004",
        "title": "LLM Fine-Tuning Researcher",
        "company": "AI Research Lab",
        "description": (
            "Research-focused role working on PEFT techniques (LoRA, QLoRA). "
            "Fine-tune large language models for domain adaptation. "
            "Publish research and collaborate with the open-source community."
        ),
        "required_skills": ["python", "huggingface", "peft", "lora", "fine-tuning"],
        "nice_to_have_skills": ["qlora", "deepspeed", "research publications"],
        "level": "junior",
        "min_years_experience": 0.0,
        "location": "Remote",
        "remote": True,
        "salary_range": "₹10–18 LPA",
    },
]

def load():
    print("Loading sample resumes...")
    for r in RESUMES:
        resp = requests.post(f"{BASE}/resumes/index", json=r)
        print(f"  {r['resume_id']}: {resp.json().get('status','ERROR')}")

    print("\nLoading sample jobs...")
    for j in JOBS:
        resp = requests.post(f"{BASE}/jobs/index", json=j)
        print(f"  {j['job_id']}: {resp.json().get('status','ERROR')}")

    print("\nIndex stats:")
    print(json.dumps(requests.get(f"{BASE}/stats").json(), indent=2))
    print("\nDone! Visit http://localhost:8000/docs to explore.")

if __name__ == "__main__":
    load()
