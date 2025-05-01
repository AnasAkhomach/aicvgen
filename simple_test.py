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

print("Starting test")

try:
    from state_manager import JobDescriptionData
    print("Imported JobDescriptionData")
    
    # Create a test object
    job_data = JobDescriptionData(
        raw_text="Test job description",
        skills=["Python"], 
        experience_level="Senior",
        responsibilities=["Code"],
        industry_terms=["Tech"],
        company_values=["Innovation"]
    )
    print("Created JobDescriptionData object")
    
    # Test to_dict method
    print("Testing to_dict method")
    if hasattr(job_data, 'to_dict'):
        job_dict = job_data.to_dict()
        print(f"to_dict returned: {job_dict}")
        print(f"Skills from dict: {job_dict.get('skills', [])}")
    else:
        print("ERROR: JobDescriptionData has no to_dict method")
    
    # Test our fallback conversion approach
    print("\nTesting manual conversion")
    manual_dict = {
        "skills": getattr(job_data, "skills", []),
        "industry_terms": getattr(job_data, "industry_terms", [])
    }
    print(f"Manual conversion returned: {manual_dict}")
    print(f"Skills from manual dict: {manual_dict.get('skills', [])}")
    
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc() 