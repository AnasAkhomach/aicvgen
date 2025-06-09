"""Sample Job Descriptions for E2E Testing.

Provides realistic job descriptions for tech roles including:
- AI Engineer
- Data Scientist  
- Software Developer

Used for testing complete CV generation workflows.
"""

from src.models.data_models import JobDescriptionData
from typing import Dict, Any


def get_ai_engineer_job() -> JobDescriptionData:
    """Sample AI Engineer job description."""
    return JobDescriptionData(
        title="Senior AI Engineer",
        company="TechVision AI",
        description="""
        We are seeking a Senior AI Engineer to join our cutting-edge AI research and development team. 
        You will be responsible for designing, implementing, and deploying machine learning models 
        and AI systems that solve complex business problems. This role requires expertise in deep 
        learning, natural language processing, and MLOps practices.
        
        Our team works on exciting projects including computer vision, conversational AI, and 
        predictive analytics. You'll collaborate with cross-functional teams to bring AI solutions 
        from research to production at scale.
        """,
        requirements=[
            "Master's degree in Computer Science, AI, or related field",
            "5+ years of experience in machine learning and AI development",
            "Strong proficiency in Python and ML frameworks (TensorFlow, PyTorch)",
            "Experience with cloud platforms (AWS, GCP, Azure) and MLOps tools",
            "Knowledge of deep learning architectures and neural networks",
            "Experience with NLP libraries and transformer models",
            "Familiarity with containerization (Docker, Kubernetes)",
            "Strong mathematical background in statistics and linear algebra",
            "Experience with version control and collaborative development",
            "Excellent problem-solving and analytical skills"
        ],
        responsibilities=[
            "Design and implement machine learning models for various business applications",
            "Develop and maintain AI pipelines from data ingestion to model deployment",
            "Collaborate with data scientists to translate research into production systems",
            "Optimize model performance and ensure scalability of AI solutions",
            "Implement MLOps best practices for model versioning and monitoring",
            "Conduct code reviews and mentor junior team members",
            "Stay current with latest AI research and industry best practices",
            "Work with stakeholders to understand business requirements and constraints",
            "Document technical specifications and maintain system architecture",
            "Participate in agile development processes and sprint planning"
        ],
        skills=[
            "Python", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
            "AWS", "GCP", "Azure", "Docker", "Kubernetes", "MLflow", "Kubeflow",
            "NLP", "Computer Vision", "Deep Learning", "Neural Networks",
            "SQL", "NoSQL", "Git", "CI/CD", "REST APIs", "Microservices",
            "Statistics", "Linear Algebra", "Data Visualization", "Jupyter"
        ]
    )


def get_data_scientist_job() -> JobDescriptionData:
    """Sample Data Scientist job description."""
    return JobDescriptionData(
        title="Senior Data Scientist",
        company="DataDriven Analytics",
        description="""
        Join our analytics team as a Senior Data Scientist where you'll drive data-driven 
        decision making across the organization. You'll work with large datasets to uncover 
        insights, build predictive models, and create data products that directly impact 
        business outcomes.
        
        This role involves end-to-end data science projects from hypothesis formation to 
        model deployment. You'll collaborate with business stakeholders, engineers, and 
        product teams to solve complex analytical challenges using statistical methods 
        and machine learning techniques.
        """,
        requirements=[
            "PhD or Master's in Statistics, Mathematics, Computer Science, or related field",
            "4+ years of experience in data science and analytics",
            "Expert-level proficiency in Python and R for data analysis",
            "Strong background in statistics, hypothesis testing, and experimental design",
            "Experience with machine learning algorithms and model evaluation",
            "Proficiency in SQL and database management systems",
            "Experience with data visualization tools (Tableau, Power BI, or similar)",
            "Knowledge of big data technologies (Spark, Hadoop, or cloud equivalents)",
            "Experience with A/B testing and causal inference methods",
            "Strong communication skills for presenting findings to stakeholders"
        ],
        responsibilities=[
            "Analyze large datasets to identify trends, patterns, and business insights",
            "Design and implement predictive models for various business use cases",
            "Conduct statistical analysis and hypothesis testing for business experiments",
            "Create data visualizations and dashboards for stakeholder consumption",
            "Collaborate with engineering teams to deploy models into production",
            "Design and analyze A/B tests to measure product and feature performance",
            "Develop data products and automated reporting solutions",
            "Mentor junior data scientists and provide technical guidance",
            "Present findings and recommendations to executive leadership",
            "Ensure data quality and implement best practices for data governance"
        ],
        skills=[
            "Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn",
            "Tableau", "Power BI", "Jupyter", "Apache Spark", "Hadoop", "AWS", "GCP",
            "Statistics", "Machine Learning", "Deep Learning", "Time Series Analysis",
            "A/B Testing", "Causal Inference", "Experimental Design", "Hypothesis Testing",
            "Data Visualization", "ETL", "Data Warehousing", "Git", "Docker"
        ]
    )


def get_software_developer_job() -> JobDescriptionData:
    """Sample Software Developer job description."""
    return JobDescriptionData(
        title="Full Stack Software Developer",
        company="InnovateTech Solutions",
        description="""
        We're looking for a talented Full Stack Software Developer to join our dynamic 
        development team. You'll be responsible for building scalable web applications 
        and APIs that serve millions of users. This role offers the opportunity to work 
        with modern technologies and contribute to all aspects of the software development lifecycle.
        
        You'll collaborate with product managers, designers, and other developers to 
        create user-friendly applications that solve real-world problems. We value 
        clean code, test-driven development, and continuous learning.
        """,
        requirements=[
            "Bachelor's degree in Computer Science or equivalent experience",
            "3+ years of experience in full-stack web development",
            "Strong proficiency in JavaScript/TypeScript and modern frameworks",
            "Experience with React, Vue.js, or Angular for frontend development",
            "Backend development experience with Node.js, Python, or Java",
            "Knowledge of database design and SQL/NoSQL databases",
            "Experience with RESTful API design and development",
            "Familiarity with cloud platforms and containerization",
            "Understanding of software testing principles and practices",
            "Experience with version control systems (Git) and agile methodologies"
        ],
        responsibilities=[
            "Develop and maintain web applications using modern frontend frameworks",
            "Build robust backend APIs and microservices",
            "Design and implement database schemas and optimize queries",
            "Write comprehensive tests to ensure code quality and reliability",
            "Collaborate with UX/UI designers to implement responsive designs",
            "Participate in code reviews and maintain coding standards",
            "Debug and troubleshoot issues across the full technology stack",
            "Optimize application performance and ensure scalability",
            "Deploy applications using CI/CD pipelines and cloud infrastructure",
            "Stay updated with emerging technologies and industry best practices"
        ],
        skills=[
            "JavaScript", "TypeScript", "React", "Vue.js", "Angular", "HTML5", "CSS3",
            "Node.js", "Python", "Java", "Express.js", "Django", "Spring Boot",
            "PostgreSQL", "MongoDB", "Redis", "MySQL", "REST APIs", "GraphQL",
            "AWS", "Docker", "Kubernetes", "Git", "CI/CD", "Jenkins", "Jest",
            "Webpack", "Babel", "SASS", "Bootstrap", "Material-UI", "Agile", "Scrum"
        ]
    )


def get_all_sample_jobs() -> Dict[str, JobDescriptionData]:
    """Get all sample job descriptions for testing."""
    return {
        "ai_engineer": get_ai_engineer_job(),
        "data_scientist": get_data_scientist_job(),
        "software_developer": get_software_developer_job()
    }


def get_job_by_role(role: str) -> JobDescriptionData:
    """Get a specific job description by role name."""
    jobs = get_all_sample_jobs()
    if role not in jobs:
        raise ValueError(f"Unknown role: {role}. Available roles: {list(jobs.keys())}")
    return jobs[role]