import google.generativeai as genai
import os
import sys
import time
import concurrent.futures
import threading

class LLM:
    """
    A wrapper class for the Google Generative AI model (Gemini).
    """
    
    def __init__(self, timeout=30):
        """
        Initialize the LLM with API key and model configuration.
        
        Args:
            timeout: Maximum time in seconds to wait for LLM response
        """
        # Configure API key
        api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyCUi6CPAsGcYkpatm6cil8_XMCfwUCmV3I")
        
        # Initialize the model
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel('gemini-2.0-flash')
        self.timeout = timeout
        
    def _generate_with_timeout(self, prompt):
        """
        Generate content with a timeout using ThreadPoolExecutor.
        
        Args:
            prompt: Text prompt to send to the model
            
        Returns:
            Generated text response or raises TimeoutError
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.llm.generate_content, prompt)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                # Try to cancel if possible
                future.cancel()
                raise TimeoutError(f"LLM request timed out after {self.timeout} seconds")
        
    def generate_content(self, prompt):
        """
        Generate content using the Gemini model with timeout and error handling.
        
        Args:
            prompt: Text prompt to send to the model
            
        Returns:
            Generated text response or a fallback message for errors
            
        Note:
            Unlike the original implementation, this version doesn't raise exceptions
            but returns a fallback message indicating the error
        """
        # Normal operation with timeout
        try:
            start_time = time.time()
            print(f"Sending request to LLM at {time.strftime('%H:%M:%S')}")
            
            # Use the timeout version
            response = self._generate_with_timeout(prompt)
            
            elapsed = time.time() - start_time
            print(f"LLM response received in {elapsed:.2f} seconds")
            
            if not hasattr(response, 'text') or response.text is None:
                return "LLM returned an empty or invalid response"
            return response.text
            
        except TimeoutError as e:
            print(f"LLM request timed out: {e}")
            return f"The AI model is taking too long to respond. Please try again later or use a simpler query."
            
        except Exception as e:
            print(f"Error in LLM.generate_content: {type(e).__name__}: {e}")
            return f"The AI model encountered an issue: {type(e).__name__}. Please try again later."