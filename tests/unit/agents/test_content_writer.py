import logging
import os
import sys
import json
from llm import LLM
from content_writer_agent import ContentWriterAgent, PromptLoader
from tools_agent import ToolsAgent
from state_manager import StructuredCV, AgentIO, Section, Subsection, Item, ItemStatus, ItemType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_content_writer.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

class VerbosePromptLoader(PromptLoader):
    """Extended PromptLoader that logs when prompts are loaded and used"""
    
    def load_prompt(self, prompt_name):
        """Override load_prompt to add logging"""
        print(f"Loading prompt: {prompt_name}")
        prompt_text = super().load_prompt(prompt_name)
        print(f"Prompt content (first 200 chars): {prompt_text[:200].replace('{', '{{').replace('}', '}}')}")
        return prompt_text

# Helper to hook LLM calls
def hook_llm(llm_instance):
    original_generate = llm_instance.generate_content
    
    def logged_generate(prompt, **kwargs):
        print(f"\nSending LLM prompt ({len(prompt)} chars):")
        print(f"Preview: {prompt[:200]}...")
        
        response = original_generate(prompt, **kwargs)
        
        print(f"Received LLM response ({len(response)} chars):")
        print(f"Preview: {response[:200]}...")
        
        return response
    
    llm_instance.generate_content = logged_generate
    return llm_instance

def load_structured_cv_from_file():
    """Load a StructuredCV from the pipeline_test_result.json file"""
    if os.path.exists("pipeline_test_result.json"):
        with open("pipeline_test_result.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if "structured_cv" in data:
                # Create a simplified version manually for testing
                cv = StructuredCV()
                
                # Add a section for summary
                summary_section = Section(
                    name="Executive Summary",
                    content_type="DYNAMIC",
                    order=0
                )
                summary_item = Item(
                    content="",
                    status=ItemStatus.TO_REGENERATE,
                    item_type=ItemType.SUMMARY_PARAGRAPH
                )
                summary_section.items.append(summary_item)
                cv.sections.append(summary_section)
                
                # Add a section for key qualifications
                quals_section = Section(
                    name="Key Qualifications",
                    content_type="DYNAMIC",
                    order=1
                )
                for _ in range(5):  # Add 5 empty quals
                    qual_item = Item(
                        content="",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.KEY_QUAL
                    )
                    quals_section.items.append(qual_item)
                cv.sections.append(quals_section)
                
                # Add a section for experience
                exp_section = Section(
                    name="Professional Experience",
                    content_type="DYNAMIC",
                    order=2
                )
                
                # Add some subsections for experience
                subsection1 = Subsection(
                    name="Retail Sales Experience"
                )
                for _ in range(3):  # Add 3 empty bullet points
                    bullet_item = Item(
                        content="",
                        status=ItemStatus.TO_REGENERATE,
                        item_type=ItemType.BULLET_POINT
                    )
                    subsection1.items.append(bullet_item)
                exp_section.subsections.append(subsection1)
                
                cv.sections.append(exp_section)
                
                return cv
    
    # Return a default empty CV if no file found or error
    return StructuredCV()

# Create a simple dictionary-based job description data
# Instead of using JobDescriptionData class which might have compatibility issues
def create_job_description_dict():
    """Create a dictionary that mimics JobDescriptionData for compatibility"""
    return {
        "raw_text": "Retail Salesperson position with responsibilities for customer service, checkout, and stocking shelves.",
        "skills": ["Customer service", "Communication", "Teamwork", "Time flexibility"],
        "experience_level": "Entry-Level",
        "responsibilities": ["Stocking shelves", "Operating checkout", "Providing customer advice"],
        "industry_terms": ["Retail", "Food retail"],
        "company_values": ["Stability", "Teamwork"]
    }

def main():
    print("\n" + "="*70)
    print("TESTING CONTENT WRITER AGENT DIRECTLY")
    print("="*70)
    
    # Initialize LLM with logging
    print("Initializing LLM...")
    llm = LLM()
    llm = hook_llm(llm)
    
    # Initialize PromptLoader with logging
    print("Initializing PromptLoader...")
    prompt_loader = VerbosePromptLoader()
    
    # Initialize tools agent
    print("Initializing ToolsAgent...")
    tools_agent = ToolsAgent(
        name="ToolsAgent", 
        description="Agent for providing content processing tools."
    )
    
    # Initialize content writer agent
    print("Initializing ContentWriterAgent...")
    content_writer_agent = ContentWriterAgent(
        name="ContentWriterAgent", 
        description="Agent for generating tailored CV content.", 
        llm=llm, 
        tools_agent=tools_agent
    )
    
    # Inject our verbose prompt loader
    content_writer_agent.prompt_loader = prompt_loader
    
    # Load test data
    print("Loading test data...")
    structured_cv = load_structured_cv_from_file()
    job_description_data = create_job_description_dict()
    
    # Prepare research results
    research_results = {
        "job_requirements_analysis": {
            "core_technical_skills": ["Customer service", "Checkout operation", "Stocking shelves"],
            "soft_skills": ["Communication", "Teamwork", "Time management"],
            "working_environment": "Retail store with focus on customer interaction"
        },
        "key_matches": {
            "skills": [
                {"content": "Customer service experience", "relevance": 0.85},
                {"content": "Team collaboration", "relevance": 0.78}
            ],
            "responsibilities": [
                {"content": "Maintained product displays and restocked shelves", "relevance": 0.92},
                {"content": "Operated cash register and processed transactions", "relevance": 0.89}
            ]
        }
    }
    
    # Run the content writer agent
    print("\n" + "="*70)
    print("RUNNING CONTENT WRITER AGENT")
    print("="*70)
    
    input_data = {
        "job_description_data": job_description_data,
        "structured_cv": structured_cv,
        "research_results": research_results,
        "regenerate_item_ids": []  # Regenerate all
    }
    
    result = content_writer_agent.run(input_data)
    
    # Save the result
    print("\n" + "="*70)
    print("CONTENT WRITER AGENT RESULTS")
    print("="*70)
    
    if result:
        print(f"Generated {len(result.sections)} sections")
        
        # Print details of what was generated
        for section in result.sections:
            print(f"\nSection: {section.name}")
            
            # Print items in the section
            if section.items:
                print(f"  Items ({len(section.items)}):")
                for item in section.items:
                    print(f"    - [{item.status.value}] {item.content[:50]}...")
            
            # Print subsections
            if section.subsections:
                print(f"  Subsections ({len(section.subsections)}):")
                for subsection in section.subsections:
                    print(f"    * {subsection.name}")
                    if subsection.items:
                        print(f"      Items ({len(subsection.items)}):")
                        for item in subsection.items:
                            print(f"        - [{item.status.value}] {item.content[:50]}...")
    else:
        print("No result returned from content writer agent")
    
    # Save result to file
    result_file = "content_writer_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict() if result else {}, f, indent=2)
    
    print(f"\nResults saved to {result_file}")
    print("="*70)

if __name__ == "__main__":
    main() 