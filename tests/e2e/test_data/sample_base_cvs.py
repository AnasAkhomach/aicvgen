"""Sample Base CVs for E2E Testing.

Provides realistic base CVs with various experience levels and formats:
- Junior level (1-2 years experience)
- Mid-level (3-5 years experience)  
- Senior level (5+ years experience)
- Different formatting styles

Used for testing CV tailoring across different candidate profiles.
"""

from typing import Dict, Any


def get_junior_developer_cv() -> str:
    """Sample CV for a junior developer (1-2 years experience)."""
    return """
# Alex Chen
**Email:** alex.chen@email.com | **Phone:** (555) 123-4567 | **Location:** San Francisco, CA  
**LinkedIn:** linkedin.com/in/alexchen | **GitHub:** github.com/alexchen

## Professional Summary
Motivated junior software developer with 2 years of experience building web applications using modern JavaScript frameworks. Passionate about clean code, user experience, and continuous learning. Strong foundation in computer science fundamentals with hands-on experience in full-stack development.

## Professional Experience

### Junior Software Developer | StartupTech Inc. | Jan 2022 - Present
- Developed responsive web applications using React and Node.js
- Collaborated with senior developers to implement new features and bug fixes
- Participated in code reviews and followed agile development practices
- Built RESTful APIs using Express.js and integrated with PostgreSQL databases
- Wrote unit tests using Jest and maintained 85% code coverage
- Assisted in deployment processes using Docker and AWS services

### Software Development Intern | TechCorp | Jun 2021 - Dec 2021
- Built internal tools using Python and Flask framework
- Worked on data visualization dashboards using Chart.js and D3.js
- Participated in daily standups and sprint planning meetings
- Learned version control best practices using Git and GitHub
- Contributed to documentation and technical specifications

## Education

### Bachelor of Science in Computer Science | University of California, Berkeley | 2021
- **Relevant Coursework:** Data Structures, Algorithms, Database Systems, Software Engineering
- **GPA:** 3.7/4.0
- **Senior Project:** Built a task management web application using MERN stack

## Technical Skills

### Programming Languages
- **Proficient:** JavaScript, Python, HTML5, CSS3
- **Familiar:** Java, SQL, TypeScript

### Frameworks & Libraries
- **Frontend:** React, Vue.js, Bootstrap, Material-UI
- **Backend:** Node.js, Express.js, Flask
- **Testing:** Jest, Pytest

### Tools & Technologies
- **Databases:** PostgreSQL, MongoDB
- **Cloud:** AWS (EC2, S3, RDS)
- **DevOps:** Docker, Git, GitHub Actions
- **Other:** REST APIs, JSON, Agile/Scrum

## Projects

### Personal Finance Tracker | 2021
- Built a full-stack web application for tracking personal expenses
- **Technologies:** React, Node.js, Express.js, PostgreSQL
- **Features:** User authentication, data visualization, expense categorization
- **GitHub:** github.com/alexchen/finance-tracker

### Weather Dashboard | 2021
- Created a responsive weather application with location-based forecasts
- **Technologies:** JavaScript, HTML5, CSS3, OpenWeather API
- **Features:** Real-time weather data, 5-day forecast, geolocation
- **Live Demo:** alexchen.github.io/weather-dashboard

## Certifications
- AWS Certified Cloud Practitioner (2022)
- FreeCodeCamp Responsive Web Design (2021)

## Additional Information
- **Languages:** English (Native), Mandarin (Conversational)
- **Interests:** Open source contributions, hackathons, tech meetups
- **Volunteer Work:** Code mentor for local coding bootcamp (2022-Present)
"""


def get_mid_level_engineer_cv() -> str:
    """Sample CV for a mid-level engineer (3-5 years experience)."""
    return """
# Sarah Johnson
## Senior Software Engineer

**Contact Information**
- Email: sarah.johnson@email.com
- Phone: (555) 987-6543
- Location: Austin, TX
- LinkedIn: linkedin.com/in/sarahjohnson
- GitHub: github.com/sarahjohnson

---

## Professional Summary

Experienced software engineer with 5 years of expertise in building scalable web applications and distributed systems. Proven track record of leading technical initiatives, mentoring junior developers, and delivering high-quality software solutions. Strong background in cloud architecture, microservices, and DevOps practices.

---

## Professional Experience

**Senior Software Engineer** | *CloudScale Technologies* | *Mar 2021 - Present*
- Lead development of microservices architecture serving 2M+ daily active users
- Designed and implemented RESTful APIs using Python/Django and Node.js
- Reduced system latency by 40% through database optimization and caching strategies
- Mentored 3 junior developers and conducted technical interviews
- Implemented CI/CD pipelines using Jenkins and Docker, reducing deployment time by 60%
- Collaborated with product managers and designers in agile development environment

**Software Engineer** | *DataFlow Solutions* | *Jun 2019 - Feb 2021*
- Developed data processing pipelines handling 100GB+ daily data volume
- Built real-time analytics dashboard using React and D3.js
- Integrated third-party APIs and implemented OAuth 2.0 authentication
- Optimized SQL queries and database schemas, improving performance by 35%
- Participated in on-call rotation and incident response procedures
- Led migration from monolithic to microservices architecture

**Junior Software Developer** | *TechStart Inc.* | *Aug 2018 - May 2019*
- Developed customer-facing web applications using React and Redux
- Implemented automated testing suites with 90%+ code coverage
- Collaborated with QA team to identify and resolve software defects
- Contributed to technical documentation and code review processes
- Participated in agile ceremonies and sprint planning

---

## Technical Skills

**Programming Languages**
- Expert: Python, JavaScript, TypeScript, SQL
- Proficient: Java, Go, HTML5, CSS3
- Familiar: C++, Rust

**Frameworks & Libraries**
- Backend: Django, Flask, Node.js, Express.js, Spring Boot
- Frontend: React, Redux, Vue.js, Angular
- Testing: Jest, Pytest, Selenium, Cypress

**Cloud & Infrastructure**
- AWS: EC2, S3, RDS, Lambda, CloudFormation, EKS
- Azure: App Service, Cosmos DB, Functions
- DevOps: Docker, Kubernetes, Jenkins, Terraform
- Monitoring: Prometheus, Grafana, ELK Stack

**Databases**
- Relational: PostgreSQL, MySQL, SQL Server
- NoSQL: MongoDB, Redis, Elasticsearch
- Data Warehousing: Snowflake, BigQuery

---

## Education

**Master of Science in Computer Science** | *University of Texas at Austin* | *2018*
- Specialization: Distributed Systems and Machine Learning
- Thesis: "Optimizing Microservices Communication Patterns"
- GPA: 3.8/4.0

**Bachelor of Science in Software Engineering** | *Texas A&M University* | *2016*
- Magna Cum Laude, GPA: 3.9/4.0
- President of Computer Science Student Association

---

## Key Projects

**E-commerce Platform Redesign** | *CloudScale Technologies* | *2022*
- Led technical design for platform handling $50M+ annual revenue
- Implemented event-driven architecture using Apache Kafka
- Reduced page load times by 50% through performance optimization
- Technologies: Python, React, PostgreSQL, Redis, AWS

**Real-time Analytics Engine** | *DataFlow Solutions* | *2020*
- Built streaming data pipeline processing 1M+ events per hour
- Designed scalable architecture using Apache Spark and Kafka
- Created interactive dashboards for business intelligence
- Technologies: Python, Apache Spark, Kafka, Elasticsearch, React

---

## Certifications & Awards

- AWS Certified Solutions Architect - Professional (2022)
- Certified Kubernetes Administrator (CKA) (2021)
- Google Cloud Professional Cloud Architect (2020)
- "Outstanding Technical Contribution" Award - CloudScale Technologies (2022)
- "Innovation Award" - DataFlow Solutions (2020)

---

## Publications & Speaking

- "Microservices Design Patterns" - Tech Conference Austin (2022)
- "Building Scalable Data Pipelines" - Medium Article (2021)
- "Best Practices for API Design" - Company Tech Blog (2020)

---

## Additional Information

- **Languages:** English (Native), Spanish (Intermediate)
- **Open Source:** Contributor to Apache Kafka and React projects
- **Community:** Organizer of Austin Women in Tech meetup
- **Interests:** Machine learning, blockchain technology, hiking
"""


def get_senior_architect_cv() -> str:
    """Sample CV for a senior architect/lead (8+ years experience)."""
    return """
# Dr. Michael Rodriguez
## Principal Software Architect & Engineering Lead

📧 michael.rodriguez@email.com | 📱 (555) 456-7890 | 📍 Seattle, WA  
🔗 linkedin.com/in/michaelrodriguez | 💻 github.com/mrodriguez

---

## Executive Summary

Principal Software Architect with 12+ years of experience designing and implementing large-scale distributed systems. Proven leader in driving technical strategy, architecture decisions, and engineering excellence across multiple organizations. Expert in cloud-native architectures, microservices, and AI/ML systems with a track record of delivering solutions that scale to millions of users.

**Core Competencies:** System Architecture • Technical Leadership • Cloud Computing • AI/ML Engineering • DevOps • Team Building

---

## Professional Experience

### Principal Software Architect | **Amazon Web Services** | *Jan 2020 - Present*

**Technical Leadership & Strategy**
- Lead architecture design for AWS AI/ML services used by 100,000+ customers globally
- Define technical roadmap and standards for distributed machine learning platforms
- Mentor 15+ senior engineers and establish engineering best practices across teams
- Drive cross-functional collaboration with product, research, and business teams

**Key Achievements**
- Architected serverless ML inference platform reducing customer costs by 60%
- Led migration of legacy systems to cloud-native architecture (500+ microservices)
- Established center of excellence for AI/ML engineering practices
- Designed fault-tolerant systems achieving 99.99% uptime SLA

**Technologies:** Python, Java, Go, AWS (Lambda, EKS, SageMaker), Kubernetes, Terraform

### Senior Engineering Manager | **Netflix** | *Mar 2017 - Dec 2019*

**Engineering Leadership**
- Managed 25+ engineers across 4 teams building content recommendation systems
- Scaled recommendation engine to serve 200M+ users with sub-100ms latency
- Implemented A/B testing framework processing 1B+ daily events
- Established data-driven culture and metrics-focused development practices

**Technical Contributions**
- Designed real-time ML pipeline using Apache Kafka and Apache Spark
- Led adoption of containerization and service mesh (Istio) across organization
- Implemented chaos engineering practices improving system resilience
- Architected multi-region deployment strategy for global content delivery

**Technologies:** Python, Scala, Java, Apache Spark, Kafka, Cassandra, AWS, Kubernetes

### Staff Software Engineer | **Google** | *Jun 2014 - Feb 2017*

**Technical Innovation**
- Core contributor to Google Cloud ML Engine (now AI Platform)
- Designed distributed training infrastructure for deep learning models
- Built auto-scaling systems handling variable ML workloads
- Collaborated with research teams to productionize cutting-edge ML algorithms

**Impact & Recognition**
- Patents: 3 granted patents in distributed machine learning systems
- Publications: 5 peer-reviewed papers in top-tier conferences (ICML, NeurIPS)
- Awards: "Technical Excellence Award" (2016), "Innovation Award" (2015)

**Technologies:** Python, C++, TensorFlow, Google Cloud Platform, Kubernetes, gRPC

### Senior Software Engineer | **Microsoft** | *Aug 2011 - May 2014*

**Product Development**
- Led development of Azure Machine Learning Studio
- Built visual ML workflow designer used by 50,000+ data scientists
- Implemented scalable compute engine for ML model training and inference
- Established DevOps practices and automated testing frameworks

**Technologies:** C#, .NET, Azure, SQL Server, JavaScript, Angular

---

## Education & Credentials

**Ph.D. in Computer Science** | *Stanford University* | *2011*
- Dissertation: "Distributed Algorithms for Large-Scale Machine Learning"
- Advisor: Prof. Andrew Ng
- Research Focus: Distributed systems, machine learning, optimization

**M.S. in Computer Science** | *Stanford University* | *2008*
- Specialization: Artificial Intelligence and Systems
- GPA: 3.9/4.0

**B.S. in Computer Engineering** | *UC Berkeley* | *2006*
- Summa Cum Laude, Phi Beta Kappa
- Outstanding Senior Award in Computer Engineering

---

## Technical Expertise

### **Programming Languages**
- **Expert:** Python, Java, Go, SQL, JavaScript/TypeScript
- **Advanced:** C++, Scala, C#, R, MATLAB
- **Familiar:** Rust, Swift, Kotlin

### **Cloud Platforms & Infrastructure**
- **AWS:** EC2, Lambda, EKS, SageMaker, S3, RDS, CloudFormation, CDK
- **Google Cloud:** GKE, AI Platform, BigQuery, Pub/Sub, Cloud Functions
- **Azure:** AKS, ML Studio, Cosmos DB, Functions, Service Fabric
- **DevOps:** Kubernetes, Docker, Terraform, Helm, Jenkins, GitLab CI/CD

### **Data & ML Technologies**
- **Frameworks:** TensorFlow, PyTorch, Scikit-learn, XGBoost, Spark MLlib
- **Data Processing:** Apache Spark, Kafka, Airflow, Beam, Flink
- **Databases:** PostgreSQL, MongoDB, Cassandra, Redis, Elasticsearch
- **Monitoring:** Prometheus, Grafana, Datadog, New Relic, ELK Stack

### **Architecture Patterns**
- Microservices, Event-Driven Architecture, CQRS, Domain-Driven Design
- Serverless Computing, Container Orchestration, Service Mesh
- Data Lakes, Lambda Architecture, Kappa Architecture

---

## Key Projects & Achievements

### **AWS SageMaker Inference Optimization** | *2021-2022*
- **Challenge:** Reduce ML model inference costs while maintaining performance
- **Solution:** Designed auto-scaling serverless inference platform
- **Impact:** 60% cost reduction, 99.9% availability, 50ms p95 latency
- **Technologies:** AWS Lambda, EKS, API Gateway, CloudWatch

### **Netflix Global Recommendation Engine** | *2018-2019*
- **Challenge:** Scale recommendation system to 200M+ global users
- **Solution:** Multi-region, real-time ML pipeline with A/B testing
- **Impact:** 15% increase in user engagement, 99.99% uptime
- **Technologies:** Kafka, Spark, Cassandra, Kubernetes, Istio

### **Google Cloud ML Engine** | *2015-2016*
- **Challenge:** Democratize machine learning for enterprise customers
- **Solution:** Managed ML platform with distributed training capabilities
- **Impact:** 10,000+ customers, $100M+ revenue contribution
- **Technologies:** TensorFlow, Kubernetes, gRPC, Google Cloud

---

## Publications & Patents

### **Patents**
1. "Distributed Training of Machine Learning Models" - US Patent 10,123,456 (2018)
2. "Auto-scaling Infrastructure for ML Workloads" - US Patent 10,234,567 (2019)
3. "Fault-tolerant Distributed Computing Systems" - US Patent 10,345,678 (2020)

### **Selected Publications**
1. "Scalable Distributed Machine Learning with Parameter Servers" - ICML 2016
2. "Efficient Resource Management for ML Workloads in the Cloud" - NeurIPS 2017
3. "Real-time Recommendation Systems at Scale" - KDD 2019
4. "Chaos Engineering for ML Systems" - MLSys 2021

---

## Professional Certifications

- **AWS Certified Solutions Architect - Professional** (2023)
- **Google Cloud Professional Cloud Architect** (2022)
- **Certified Kubernetes Administrator (CKA)** (2021)
- **Azure Solutions Architect Expert** (2020)
- **TensorFlow Developer Certificate** (2020)

---

## Leadership & Community

### **Speaking & Conferences**
- **Keynote Speaker:** "The Future of ML Infrastructure" - MLOps World 2023
- **Technical Talk:** "Building Resilient ML Systems" - KubeCon 2022
- **Panel Discussion:** "AI Ethics in Production" - AI Summit 2021

### **Open Source Contributions**
- **TensorFlow:** Core contributor, 50+ merged PRs
- **Kubernetes:** SIG-ML lead, ML workload scheduling improvements
- **Apache Spark:** MLlib contributor, distributed training optimizations

### **Mentorship & Teaching**
- **Stanford University:** Guest lecturer for CS229 (Machine Learning)
- **Industry Mentorship:** 20+ engineers mentored to senior/staff levels
- **Diversity & Inclusion:** Co-founder of "Engineers for Equity" initiative

---

## Awards & Recognition

- **"Technical Visionary Award"** - AWS (2022)
- **"Engineering Excellence Award"** - Netflix (2019)
- **"Innovation Impact Award"** - Google (2016)
- **"40 Under 40 in Tech"** - TechCrunch (2020)
- **"Top ML Engineer"** - Towards Data Science (2021)

---

## Additional Information

**Languages:** English (Native), Spanish (Fluent), Portuguese (Conversational)  
**Security Clearance:** Secret (DoD) - Active  
**Interests:** AI Ethics, Quantum Computing, Rock Climbing, Photography  
**Volunteer Work:** STEM Education Advocate, Code.org Volunteer
"""


def get_all_sample_cvs() -> Dict[str, str]:
    """Get all sample base CVs for testing."""
    return {
        "junior_developer": get_junior_developer_cv(),
        "mid_level_engineer": get_mid_level_engineer_cv(),
        "senior_architect": get_senior_architect_cv()
    }


def get_cv_by_level(level: str) -> str:
    """Get a specific CV by experience level."""
    cvs = get_all_sample_cvs()
    if level not in cvs:
        raise ValueError(f"Unknown level: {level}. Available levels: {list(cvs.keys())}")
    return cvs[level]