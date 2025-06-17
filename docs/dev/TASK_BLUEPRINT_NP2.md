---

Absolutely. We will now proceed to the Priority 2 tasks, which are focused on improving the system's resilience. Here is the blueprint for the first, and arguably most impactful, of these tasks.

### **TASK\_BLUEPRINT.md**

---

### **Refactoring Task 4: Centralize and Fortify LLM Output Parsing**

**1. Task/Feature Addressed**

Multiple agents currently parse loosely structured Markdown or text returned by the LLM. This approach is highly brittle; minor changes in the LLM's output format can break the parsing logic, leading to runtime errors and data corruption.

This refactoring will enforce a robust, explicit contract between the agents and the LLM. We will mandate that the LLM returns structured JSON for all content generation tasks. This JSON will then be validated against specific Pydantic models, ensuring the data is in the expected format before it is used, thus dramatically increasing system reliability.

**2. Affected Component(s)**

* **`data/prompts/*.md`**: Prompts used for content generation will be updated to request JSON output. (e.g., `job_description_parsing_prompt.md`, `resume_role_prompt.md`).
* **`src/models/validation_schemas.py`**: Will be populated with new Pydantic models, each defining the expected JSON schema for a specific LLM task.
* **`src/agents/*.py`**: Agents that parse LLM responses will have their parsing logic completely refactored to use the new JSON-based approach (e.g., `ParserAgent`, `EnhancedContentWriterAgent`).
* **`src/utils/exceptions.py`**: The existing `LLMResponseParsingError` will be leveraged.

**3. Pydantic Model Changes**

This is a core component of the task. New Pydantic models will be created to serve as the validation schemas for LLM responses.

* **Add the following models to `src/models/validation_schemas.py`:**

```python
# In src/models/validation_schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class LLMJobDescriptionOutput(BaseModel):
    """Schema for validating the JSON output from the job description parsing LLM call."""
    skills: List[str] = Field(..., description="List of key skills and technologies mentioned.")
    experience_level: str = Field(..., description="Required experience level (e.g., Senior, Mid-Level).")
    responsibilities: List[str] = Field(..., description="List of key job responsibilities.")
    industry_terms: List[str] = Field(..., description="List of industry-specific terms or jargon.")
    company_values: List[str] = Field(..., description="List of company values or cultural keywords.")

class LLMRoleGenerationOutput(BaseModel):
    """Schema for validating the JSON output for generating a single resume role."""
    organization_description: Optional[str] = Field(description="A brief description of the company.")
    role_description: Optional[str] = Field(description="A brief description of the role's main purpose.")
    bullet_points: List[str] = Field(..., description="A list of 3-5 generated resume bullet points.")

# Add other validation schemas as needed for other prompts (e.g., for projects, summaries).
```

**4. Detailed Implementation Steps**

We will use the `ParserAgent`'s `parse_job_description` method as the primary example of this refactoring pattern.

**Step 1: Update the Prompt (`job_description_parsing_prompt.md`)**

The prompt must be rewritten to demand JSON output conforming to our new Pydantic schema.

* **Modify `data/prompts/job_description_parsing_prompt.md`:**

```markdown
Your task is to act as an expert job description analyzer. Extract the key information from the following job description text and provide it in a structured JSON format.

**Job Description:**
```
{{raw_text}}
```

**Instructions:**
1.  Read the job description carefully.
2.  Extract the information for each of the following keys: `skills`, `experience_level`, `responsibilities`, `industry_terms`, `company_values`.
3.  Return **ONLY a single, valid JSON object** that strictly adheres to the schema provided below. Do not include any explanatory text, markdown formatting, or anything outside of the JSON object.

**JSON Schema:**
```json
{
  "skills": ["string"],
  "experience_level": "string",
  "responsibilities": ["string"],
  "industry_terms": ["string"],
  "company_values": ["string"]
}
```
```

**Step 2: Refactor the Agent's Parsing Logic (`ParserAgent`)**

The `parse_job_description` method in `src/agents/parser_agent.py` will be completely overhauled. The old logic for regex fallbacks and fragile string manipulation will be replaced with a clean `try-except` block for JSON parsing and Pydantic validation.

* **Refactor the `parse_job_description` method in `src/agents/parser_agent.py`:**

```python
# In src/agents/parser_agent.py

# Add these imports
import json
from src.models.validation_schemas import LLMJobDescriptionOutput
from src.utils.exceptions import LLMResponseParsingError
from pydantic import ValidationError

# ... other imports

class ParserAgent(EnhancedAgentBase):
    # ... other methods ...

    async def parse_job_description(self, raw_text: str) -> JobDescriptionData:
        if not raw_text:
            logger.warning("Empty job description provided to ParserAgent.")
            return JobDescriptionData(raw_text=raw_text, skills=[], responsibilities=[], company_values=[], industry_terms=[], experience_level="N/A")

        # Load the updated prompt
        prompt_path = self.settings.get_prompt_path("job_description_parsing_prompt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        prompt = prompt_template.format(raw_text=raw_text)

        try:
            # 1. Get response from LLM
            response = await self.llm.generate_content(prompt)
            raw_response_content = response.content

            # 2. Extract the JSON block from the raw response
            # A simple but effective way to handle markdown code blocks or other noise
            json_start = raw_response_content.find('{')
            json_end = raw_response_content.rfind('}') + 1
            if json_start == -1 or json_end <= json_start:
                raise LLMResponseParsingError("No valid JSON object found in LLM response.", raw_response=raw_response_content)

            json_str = raw_response_content[json_start:json_end]

            # 3. Parse the JSON string
            parsed_data = json.loads(json_str)

            # 4. Validate with Pydantic
            validated_output = LLMJobDescriptionOutput.model_validate(parsed_data)

            # 5. Map validated data to the application's main data model
            job_data = JobDescriptionData(
                raw_text=raw_text,
                skills=validated_output.skills,
                experience_level=validated_output.experience_level,
                responsibilities=validated_output.responsibilities,
                industry_terms=validated_output.industry_terms,
                company_values=validated_output.company_values,
                status=ItemStatus.GENERATED
            )
            logger.info("Job description successfully parsed and validated using LLM-generated JSON.")
            return job_data

        except (json.JSONDecodeError, ValidationError, LLMResponseParsingError) as e:
            error_message = f"Failed to parse or validate LLM response for job description: {str(e)}"
            logger.error(error_message, exc_info=True)
            # Create a failed state object
            return JobDescriptionData(
                raw_text=raw_text,
                skills=[], responsibilities=[], company_values=[], industry_terms=[], experience_level="N/A",
                error=error_message,
                status=ItemStatus.GENERATION_FAILED
            )
        except Exception as e:
            # Handle other potential errors like LLM API failures
            error_message = f"An unexpected error occurred during job description parsing: {str(e)}"
            logger.error(error_message, exc_info=True)
            return JobDescriptionData(
                raw_text=raw_text,
                skills=[], responsibilities=[], company_values=[], industry_terms=[], experience_level="N/A",
                error=error_message,
                status=ItemStatus.GENERATION_FAILED
            )
```

**Step 3: Apply This Pattern to Other Agents**

This same pattern should be applied to all other agents that generate structured content, especially the `EnhancedContentWriterAgent`.
1.  Create a Pydantic validation schema for each distinct output type in `validation_schemas.py`.
2.  Update the corresponding prompt in `data/prompts/` to request JSON matching that schema.
3.  Replace the agent's string-parsing logic with the `try-except` block for JSON parsing and Pydantic validation.

**5. Testing Considerations**

* **Unit Tests for Parsing Logic:** Create new unit tests for the updated agent methods.
    * **Test Case 1 (Happy Path):** Provide a mock LLM response containing a valid JSON string. Assert that the method successfully parses it and returns a correctly populated `JobDescriptionData` object.
    * **Test Case 2 (Invalid JSON):** Provide a malformed JSON string. Assert that an `LLMResponseParsingError` or `JSONDecodeError` is caught and the method returns a `JobDescriptionData` object with a `GENERATION_FAILED` status and a descriptive error message.
    * **Test Case 3 (Schema Mismatch):** Provide valid JSON that is missing a required field (e.g., `skills`). Assert that a Pydantic `ValidationError` is caught and the method returns a `JobDescriptionData` object with a `GENERATION_FAILED` status.

---
Executing this blueprint will fundamentally improve the reliability of your data processing pipeline. It replaces implicit, brittle agreements with explicit, machine-verifiable contracts.

Please confirm you are ready to move to the next task in the plan: **"Implement LaTeX Character Escaping Utility."**

---

Excellent point. Providing a detailed, concrete example for the `EnhancedContentWriterAgent` is crucial to ensure the pattern is applied correctly. You are right to request explicit instructions to prevent any ambiguity.

Here is the detailed addendum to the previous blueprint, focusing specifically on how to apply the JSON parsing pattern to the `EnhancedContentWriterAgent`.

---

### **Addendum to Blueprint 4: Detailed Application for `EnhancedContentWriterAgent`**

This guide provides a step-by-step walkthrough for refactoring the `EnhancedContentWriterAgent` to use a robust JSON-based contract with the LLM. We will focus on the "Professional Experience" role generation as a primary example.

#### **Affected Components for this Example:**

* **`data/prompts/resume_role_prompt.md`**: The prompt for generating content for a work experience role.
* **`src/models/validation_schemas.py`**: The file containing the Pydantic schema for the LLM's JSON output.
* **`src/agents/enhanced_content_writer.py`**: The agent whose parsing logic will be refactored.

---

#### **Step 3.1: Define (or Confirm) the Pydantic Validation Schema**

First, we ensure the Pydantic model that defines the LLM's expected JSON structure exists in `src/models/validation_schemas.py`. We will use the `LLMRoleGenerationOutput` model previously defined.

* **File: `src/models/validation_schemas.py`**

```python
# This model should already be present from the previous step, but we confirm its structure.
from pydantic import BaseModel, Field
from typing import List, Optional

class LLMRoleGenerationOutput(BaseModel):
    """Schema for validating the JSON output for generating a single resume role."""
    organization_description: Optional[str] = Field(description="A brief description of the company.")
    role_description: Optional[str] = Field(description="A brief description of the role's main purpose.")
    bullet_points: List[str] = Field(..., description="A list of 3-5 generated resume bullet points tailored to the job description.")
```

---

#### **Step 3.2: Update the Corresponding Prompt**

Next, we update the `resume_role_prompt.md` to instruct the LLM to return JSON matching our `LLMRoleGenerationOutput` schema.

* **File: `data/prompts/resume_role_prompt.md`**

```markdown
You are an expert resume writer. Your task is to generate a professional description and a set of impactful, tailored bullet points for a specific role from a resume, based on a target job description.

**Target Job Description Context:**
- **Key Skills:** {{ target_skills }}
- **Responsibilities & Values:** {{ job_desc_summary }}

**Resume Role to Enhance:**
- **Title:** {{ role.title }}
- **Company:** {{ role.company }}
- **Existing Details:** {{ role.details }}

**Instructions:**
1.  Analyze the provided context from the job description and the resume.
2.  Generate a concise, professional description for the organization and the role.
3.  Create 3-5 powerful resume bullet points. Each bullet point should start with an action verb and highlight achievements relevant to the target job's key skills and responsibilities.
4.  Return **ONLY a single, valid JSON object** with the generated content. Do not include any text outside the JSON structure.

**JSON Schema to Follow:**
```json
{
  "organization_description": "string (A brief summary of the company)",
  "role_description": "string (A brief summary of the role)",
  "bullet_points": [
    "string (First bullet point)",
    "string (Second bullet point)",
    "string (Third bullet point)"
  ]
}
```
```

---

#### **Step 3.3: Refactor the Agent's String-Parsing Logic**

This is the most critical step. We will modify the `EnhancedContentWriterAgent`, specifically the logic that handles experience items. The old, brittle string-parsing will be replaced with the new JSON validation pattern.

* **File: `src/agents/enhanced_content_writer.py`**
* **Location:** Inside a method like `_process_single_item` or `run_async`, locate the logic for `ItemType.EXPERIENCE_ENTRY`.

**Illustrative "BEFORE" Logic (Conceptual):**

```python
# Conceptual representation of the old logic
# in EnhancedContentWriterAgent

# ... inside a method processing an item ...
if item.item_type == ItemType.EXPERIENCE_ENTRY:
    prompt = self._build_experience_prompt(item, job_data)
    response_text = await self.llm.generate_content(prompt)

    # Brittle parsing logic based on string splitting and regex
    bullet_points = []
    lines = response_text.split('\n')
    for line in lines:
        if line.strip().startswith('*'):
            bullet_points.append(line.strip('* ').strip())

    # Manually update the StructuredCV with the parsed strings
    # This is prone to errors if the format changes at all.
    self._update_cv_with_bullets(item.id, bullet_points)
```

**Refactored "AFTER" Logic (The New Pattern):**

```python
# The new, robust implementation in EnhancedContentWriterAgent
# Add these necessary imports at the top of the file:
import json
from src.models.validation_schemas import LLMRoleGenerationOutput
from src.utils.exceptions import LLMResponseParsingError
from pydantic import ValidationError

# ... inside a method processing an item, like _process_single_item ...
if item.item_type == ItemType.EXPERIENCE_ENTRY:
    try:
        # 1. Build the prompt using the *new* template
        prompt = self._build_experience_prompt(item, job_data) # This function now loads resume_role_prompt.md

        # 2. Get response from LLM
        response = await self.llm.generate_content(prompt)
        raw_response_content = response.content
        item.raw_llm_output = raw_response_content # Save for debugging

        # 3. Extract, Parse, and Validate JSON
        json_start = raw_response_content.find('{')
        json_end = raw_response_content.rfind('}') + 1
        if json_start == -1 or json_end <= json_start:
            raise LLMResponseParsingError("No valid JSON object found in LLM response for role generation.", raw_response=raw_response_content)

        json_str = raw_response_content[json_start:json_end]
        parsed_data = json.loads(json_str)
        validated_data = LLMRoleGenerationOutput.model_validate(parsed_data)

        # 4. Update the StructuredCV with validated data
        # This is a much safer and more predictable operation.
        # Find the subsection corresponding to this experience item.
        subsection = structured_cv.find_subsection_for_item(item.id)
        if subsection:
            # Clear existing bullet point items (or update them)
            subsection.items = []
            # Add new, validated bullet points as Item objects
            for bp_text in validated_data.bullet_points:
                new_item = Item(
                    content=bp_text,
                    status=ItemStatus.GENERATED,
                    item_type=ItemType.BULLET_POINT
                )
                subsection.items.append(new_item)
            # You can also store the role/org descriptions in the subsection's raw_text or a dedicated field if desired.

        item.status = ItemStatus.GENERATED
        logger.info(f"Successfully processed experience item {item.id} with validated JSON.")

    except (json.JSONDecodeError, ValidationError, LLMResponseParsingError) as e:
        error_message = f"Failed to parse or validate LLM response for item {item.id}: {str(e)}"
        logger.error(error_message, exc_info=True)
        item.status = ItemStatus.GENERATION_FAILED
        state.error_messages.append(error_message) # Update state with the error

    except Exception as e:
        error_message = f"An unexpected error occurred processing item {item.id}: {str(e)}"
        logger.error(error_message, exc_info=True)
        item.status = ItemStatus.GENERATION_FAILED
        state.error_messages.append(error_message)
```

---

#### **Step 3.4: Generalize This Pattern**

Now, apply this exact same robust pattern to the other content generation features within `EnhancedContentWriterAgent`:

* **For Project Experience:**
    1.  Create `LLMProjectGenerationOutput` in `validation_schemas.py`.
    2.  Update `side_project_prompt.md` to request JSON matching this new schema.
    3.  Refactor the agent's logic for `ItemType.PROJECT_ENTRY` to use the JSON parsing and validation pattern.

* **For Executive Summary:**
    1.  Create `LLMSummaryOutput` in `validation_schemas.py`.
    2.  Update `executive_summary_prompt.md` to request JSON.
    3.  Refactor the agent's summary generation logic.

* **For Key Qualifications:**
    1.  Create `LLMQualificationsOutput` in `validation_schemas.py`.
    2.  Update `key_qualifications_prompt.md` to request JSON.
    3.  Refactor the agent's qualifications generation logic.

By methodically replacing each instance of brittle string parsing with this explicit JSON contract, the entire content generation process will become significantly more reliable and easier to debug.

This detailed guide should leave no room for interpretation. Please confirm you are ready to proceed to the next task: **"Implement LaTeX Character Escaping Utility."**

---

Of course. Let's proceed with the next task in our refactoring plan, which addresses the stability of the PDF generation pipeline.

### **TASK\_BLUEPRINT.md**

---

### **Refactoring Task 5: Implement LaTeX Character Escaping Utility**

**1. Task/Feature Addressed**

The PDF generation pipeline, if it uses LaTeX directly or indirectly through templates, is highly vulnerable to compilation errors when dynamic data (from user input or LLM generation) contains special LaTeX characters (e.g., `&`, `%`, `_`, `#`, `\`). A single unescaped character can cause the entire PDF rendering process to fail.

This task involves creating a centralized, robust utility function to "sanitize" all dynamic text by escaping these special characters, ensuring reliable PDF generation regardless of the content.

**2. Affected Component(s)**

* **`src/utils/template_renderer.py`**: This is the ideal location to add the new, reusable `escape_latex` utility function.
* **`src/agents/formatter_agent.py`**: This agent, which is responsible for preparing the data and rendering the final document template, will be modified to import and use the new escaping utility.

**3. Pydantic Model Changes**

No changes to any Pydantic models are required for this task.

**4. Detailed Implementation Steps**

**Step 1: Create the `escape_latex` Utility Function**

We will define the core function and a helper to recursively apply it to data structures.

* **Add the following code to `src/utils/template_renderer.py`:**

```python
# In src/utils/template_renderer.py
import re
from typing import Any, Dict, List

# A mapping of special LaTeX characters to their escaped equivalents.
# The order is important, especially for the backslash.
LATEX_SPECIAL_CHAR_MAP = {
    '\\': r'\textbackslash{}',
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
}

def escape_latex(text: str) -> str:
    """
    Escapes special LaTeX characters in a given string.

    Args:
        text: The input string to sanitize.

    Returns:
        A string with special LaTeX characters replaced by their escaped counterparts.
    """
    if not isinstance(text, str):
        return text

    # Create a regex pattern to find any of the special characters
    # The backslash must be handled carefully in the regex pattern
    pattern = re.compile('([&%$#_{}])|([\\\\~^])') # Group 1 for most, Group 2 for others

    def replace(match):
        # Handle the two groups of characters
        if match.group(1):
            return LATEX_SPECIAL_CHAR_MAP[match.group(1)]
        elif match.group(2):
            return LATEX_SPECIAL_CHAR_MAP[match.group(2)]
        return match.group(0)

    # Use re.sub with a replacement function for clarity and correctness
    # A simple chained .replace() can have ordering issues.
    # A more robust regex approach handles this well.
    escaped_text = re.sub(
        '|'.join(re.escape(key) for key in sorted(LATEX_SPECIAL_CHAR_MAP, key=len, reverse=True)),
        lambda k: LATEX_SPECIAL_CHAR_MAP[k.group(0)],
        text
    )
    return escaped_text


def recursively_escape_latex(data: Any) -> Any:
    """
    Recursively traverses a data structure (dict, list) and applies LaTeX
    escaping to all string values.

    Args:
        data: The data structure (e.g., a dictionary for a Jinja2 context).

    Returns:
        The same data structure with all strings sanitized for LaTeX.
    """
    if isinstance(data, dict):
        return {k: recursively_escape_latex(v) for k, v in data.items()}
    if isinstance(data, list):
        return [recursively_escape_latex(elem) for elem in data]
    if isinstance(data, str):
        return escape_latex(data)
    # Return non-string, non-collection types as-is
    return data
```

**Step 2: Apply the Escaping Function in the `FormatterAgent`**

Before rendering the template, the `FormatterAgent` must process its context data using the new recursive utility.

* **Modify the `run_as_node` method (or a helper) in `src/agents/formatter_agent.py`:**

```python
# In src/agents/formatter_agent.py

# Add the new import
from src.utils.template_renderer import recursively_escape_latex
# ... other imports for AgentState, StructuredCV, etc.

class FormatterAgent(EnhancedAgentBase):

    # ... __init__ and other methods ...

    async def run_as_node(self, state: AgentState) -> dict:
        logger.info("FormatterAgent is running.")
        try:
            structured_cv = state.get("structured_cv")
            if not structured_cv:
                raise ValueError("StructuredCV is missing from the state.")

            # 1. Prepare the context dictionary from the StructuredCV Pydantic model
            # This converts the Pydantic object into a plain dictionary
            cv_context_dict = structured_cv.model_dump()

            # --- NEW: CRITICAL SANITIZATION STEP ---
            # Recursively escape all string values in the context dictionary.
            sanitized_cv_context = recursively_escape_latex(cv_context_dict)
            # --- END OF NEW STEP ---

            # 2. Load the Jinja2 template
            # (Assuming template_manager logic exists)
            template = self.template_manager.get_template("pdf_template.html") # or .tex

            # 3. Render the template using the SANITIZED context
            rendered_html_or_latex = template.render(cv=sanitized_cv_context)

            # 4. Generate the PDF from the rendered output
            # (Assuming pdf_generator logic exists)
            output_path = self.pdf_generator.generate(rendered_html_or_latex)

            logger.info(f"Successfully generated PDF at {output_path}")
            return {"final_output_path": output_path}

        except Exception as e:
            error_message = f"Failed during PDF formatting and generation: {str(e)}"
            logger.error(error_message, exc_info=True)
            return {"error_messages": state.get("error_messages", []) + [error_message]}

```

**5. Testing Considerations**

* **Unit Tests for the Utility (`test_template_renderer.py`):**
    * Create a new test file `tests/unit/test_template_renderer.py`.
    * **Test `escape_latex`:** Write a test that passes a string containing all special characters (e.g., `Hello _world_ & Co. 100% #1 {awesome}`) and asserts that the output is correctly escaped (e.g., `Hello \_world\_ \& Co. 100\% \#1 \{awesome\}`).
    * **Test `recursively_escape_latex`:** Write a test that passes a nested dictionary/list structure containing strings and other data types. Assert that only the strings are modified and the overall structure is preserved.

* **Integration Test for PDF Generation:**
    * In `tests/integration/test_pdf_workflow_integration.py`, create a test case that builds a `StructuredCV` object with content known to contain special characters (e.g., a company name like "AT&T", a skill like "C#", a project name like "Project_Alpha").
    * Run the `FormatterAgent` on this `StructuredCV`.
    * The test should assert that the PDF generation process completes *successfully* without raising a LaTeX compilation error and that a valid PDF file is created.

---
By implementing this utility, you will have fortified one of the most common and frustrating failure points in automated document generation pipelines.

This concludes the Priority 2 tasks. We can now proceed to the **Priority 3** tasks, which focus on general code hygiene and maintainability. The first of these is **"Refactor Hardcoded Configurations."** Please confirm when you are ready to proceed.

---