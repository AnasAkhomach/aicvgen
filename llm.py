import google.generativeai as genai

class LLM:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key="AIzaSyCUi6CPAsGcYkpatm6cil8_XMCfwUCmV3I")  # TODO: Replace with environment variable
        self.llm = genai.GenerativeModel('gemini-2.0-flash')

    def generate_content(self, prompt):
        response = self.llm.generate_content(prompt)
        return response.text