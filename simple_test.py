# simple_test.py
from parser_agent import ParserAgent
from llm import LLM

# Set up the LLM
llm = LLM(timeout=30)

# Set up parser agent
parser_agent = ParserAgent(
    name="Job Description Parser",
    description="Parses job descriptions to extract key information.",
    llm=llm
)

# Test data
job_description = "Software Engineer position at Google. Required skills: Python, Java, JavaScript, and cloud technologies. Experience level: 3-5 years. Responsibilities include developing web applications, maintaining code quality, and collaborating with team members."

# Run the parser
print("Starting job description parsing test...")
result = parser_agent.run({"job_description": job_description})
print("\nParsing result:")
print(f"Skills: {result.skills}")
print(f"Experience Level: {result.experience_level}")
print(f"Responsibilities: {result.responsibilities}")
print(f"Industry Terms: {result.industry_terms}")
print(f"Company Values: {result.company_values}")
print(f"Error: {result.error}") 