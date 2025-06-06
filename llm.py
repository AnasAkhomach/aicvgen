import google.generativeai as genai
import os
import sys
import time
import concurrent.futures
import threading
from queue import Queue
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple rate limiter to prevent exceeding API quotas.
    Enforces a maximum number of requests per minute.
    """

    def __init__(self, max_requests_per_minute=12):
        """
        Initialize the rate limiter.

        Args:
            max_requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests = max_requests_per_minute
        self.interval = 60 / max_requests_per_minute  # seconds between requests
        self.request_times = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """
        Wait if necessary to respect the rate limit.
        """
        with self.lock:
            now = time.time()

            # Remove old requests from the tracking window
            self.request_times = [t for t in self.request_times if now - t < 60]

            # If we've hit the limit, wait until we can make another request
            if len(self.request_times) >= self.max_requests:
                oldest_request = min(self.request_times)
                sleep_time = max(0, 60 - (now - oldest_request))

                if sleep_time > 0:
                    logger.info(
                        "Rate limit reached. Waiting %.2f seconds before next request",
                        sleep_time,
                    )
                    # Release the lock while sleeping
                    self.lock.release()
                    time.sleep(sleep_time)
                    # Re-acquire the lock
                    self.lock.acquire()

            # Add this request to the tracking window
            self.request_times.append(time.time())


class LLM:
    """
    A wrapper class for the Google Generative AI model (Gemini).
    """

    def __init__(self, timeout=30, max_requests_per_minute=12):
        """
        Initialize the LLM with API key and model configuration.

        Args:
            timeout: Maximum time in seconds to wait for LLM response
            max_requests_per_minute: Maximum requests allowed per minute
        """
        # Configure API key
        api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyCUi6CPAsGcYkpatm6cil8_XMCfwUCmV3I")

        # Initialize the model
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel("gemini-2.0-flash")
        self.timeout = timeout

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(max_requests_per_minute)

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
            This implementation respects rate limits and handles errors gracefully
        """
        # Apply rate limiting before making the request
        self.rate_limiter.wait_if_needed()

        # Normal operation with timeout
        try:
            # Log prompt snippet for debugging
            prompt_snippet = prompt[:200] + "..." if len(prompt) > 200 else prompt
            print(f"📤 PROMPT: {prompt_snippet}")

            start_time = time.time()
            print(f"Sending request to LLM at {time.strftime('%H:%M:%S')}")

            # Use the timeout version
            response = self._generate_with_timeout(prompt)

            elapsed = time.time() - start_time
            print(f"LLM response received in {elapsed:.2f} seconds")

            if not hasattr(response, "text") or response.text is None:
                return "LLM returned an empty or invalid response"

            # Log response snippet for debugging
            response_snippet = (
                response.text[:200] + "..." if len(response.text) > 200 else response.text
            )
            print(f"📥 RESPONSE: {response_snippet}")

            return response.text

        except TimeoutError as e:
            print(f"LLM request timed out: {e}")
            return f"The AI model is taking too long to respond. Please try again later or use a simpler query."

        except Exception as e:
            print(f"Error in LLM.generate_content: {type(e).__name__}: {e}")

            # For rate limiting errors, wait longer and suggest a solution
            if "ResourceExhausted" in str(e) or "quota" in str(e).lower() or "429" in str(e):
                retry_message = "Rate limit exceeded. Consider using a paid API key or reducing the request frequency."

                # Try to extract retry delay from error message
                import re

                retry_seconds = 60  # Default to 60 seconds
                delay_match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(e))
                if delay_match:
                    retry_seconds = int(delay_match.group(1))

                print(
                    f"Rate limit exceeded. Waiting {retry_seconds} seconds before allowing next request."
                )

                # Enforce a longer wait for the next request
                time.sleep(
                    min(retry_seconds, 5)
                )  # Wait at most 5 seconds here, the rate limiter will handle the rest

                return f"The AI model encountered a rate limit issue. {retry_message}"

            return f"The AI model encountered an issue: {type(e).__name__}. Please try again later."
