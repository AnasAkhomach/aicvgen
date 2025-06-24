

# **Root Cause Analysis and Architectural Remediation of a Critical Asynchronous Execution Failure in the aicvgen Workflow**

## **Executive Summary**

This report provides an exhaustive analysis of a critical runtime failure within the anasakhomach-aicvgen application, identified as BUG-aicvgen-002.1 The primary symptom of this failure is a

TypeError: object NoneType can't be used in 'await' expression, which consistently halts the core CV generation workflow, rendering the application non-functional.1 This issue represents a complete blocker to the system's primary operational capability.

The proximate cause of this failure has been identified as a silent initialization error within the EnhancedLLMService component, located in src/services/llm\_service.py.1 A lack of robust configuration validation, specifically for the

GEMINI\_API\_KEY, allows the service's constructor to fail without raising a fatal exception. This leads to the application's singleton service instance being assigned a None value. This None object is then injected as a dependency into downstream components, such as the ParserAgent. When the agent subsequently attempts to invoke an asynchronous method on this null service object (e.g., await self.llm\_service.generate\_content(...)), the Python runtime correctly raises a TypeError, as None is not an awaitable object.

However, this bug is not merely an isolated coding oversight. It is a direct manifestation of deeper architectural weaknesses that were previously identified in the "Codebase Analysis & Technical Debt Audit".1 The failure was allowed to manifest in a cryptic and difficult-to-debug manner due to several contributing factors: a fragile service layer that lacks fail-fast validation at startup, inconsistent error handling patterns, and a dependency management strategy (a simple singleton getter) that obscures the initial point of failure. The application's retry mechanisms further complicated the issue by attempting to recover from a fatal configuration error, thereby wasting system resources and masking the true root cause.1

Remediation has been approached in two phases. The immediate code-level solution involves modifying the EnhancedLLMService to enforce robust, fail-fast initialization, ensuring the application crashes immediately with a clear error message if the API key is not configured. This prevents the None object from ever propagating through the system. The second, more strategic phase involves a set of long-term recommendations to address the underlying architectural debt. These include centralizing resilience patterns, implementing a formal dependency injection framework, enforcing stricter quality gates in the CI pipeline, and resolving known data contract breaches between agents. Addressing these foundational issues is essential for transforming the aicvgen project from a fragile prototype into a stable and maintainable system.

## **Initial Triage and Symptomology**

The initial step in any debugging process is a thorough triage of the reported issue, combining evidence from bug reports, application logs, and error tracebacks to establish a precise understanding of the failure's symptoms and context.

### **Formal Bug Report (BUG-aicvgen-002)**

The issue is formally tracked as BUG-aicvgen-002. It is classified as a critical-severity defect because it completely blocks the application's primary user-facing workflow. The table below consolidates the key triage information from the available logs and analysis documents.

| Triage Field | Details |
| :---- | :---- |
| **Bug ID** | BUG-aicvgen-002 1 |
| **Severity** | Critical |
| **Status** | Closed (Root Cause Identified and Fix Implemented) |
| **Summary** | A persistent TypeError: object NoneType can't be used in 'await' expression occurs during LLM calls, halting the CV generation workflow after multiple retries. |
| **Error Message/Traceback** | Traceback (most recent call last): File "C:\\...\\src\\agents\\parser\_agent.py", line 628, in run\_as\_node job\_data \= await self.parse\_job\_description(...) File "C:\\...\\src\\agents\\parser\_agent.py", line 144, in \_parse\_job\_description\_with\_llm parsed\_data \= await self.\_generate\_and\_parse\_json(prompt=prompt) File "C:\\...\\src\\agents\\agent\_base.py", line 479, in \_generate\_and\_parse\_json response \= await self.llm\_service.generate\_content(...) TypeError: object NoneType can't be used in 'await' expression 1 |
| **Affected Components** | src/services/llm\_service.py, src/agents/agent\_base.py, src/agents/parser\_agent.py |

### **Analysis of Application Logs**

A chronological analysis of app-log.txt reveals the precise sequence of events leading to the failure.1 The application proceeds through a lengthy and seemingly successful initialization phase where all core components, including the

EnhancedLLMService, various agents, and the cv\_workflow\_graph, are loaded.

The workflow execution begins with a valid trace\_id and session\_id. The LangGraph orchestrator correctly invokes the first node in the sequence: parser\_node. The log messages confirm that the ParserAgent's run\_as\_node method is entered and that its initial state validation passes. The agent then attempts its primary task: parsing the job description. This is the point of failure.

The logs show a cascade of errors originating from this step:

1. An unknown\_error is recorded with the message: object NoneType can't be used in 'await' expression.  
2. The error recovery service determines an exponential\_backoff strategy, indicating a retry mechanism is in place.  
3. A message LLM generation failed appears, explicitly stating that the failure occurred after 5 retries were exhausted or a non-retriable condition was met.  
4. The failure is attributed to the ParserAgent.  
5. A full traceback is logged, pinpointing the await self.llm\_service.generate\_content(...) call in agent\_base.py as the line where the TypeError is raised.

This sequence confirms that the system is not crashing on startup but is failing during the execution of the first logical step in its workflow. The failure is directly related to an interaction with the llm\_service.

### **Interpreting the TypeError**

The error message TypeError: object NoneType can't be used in 'await' expression is fundamental to understanding the bug. In Python's asyncio framework, the await keyword is a syntactic construct used to pause the execution of a coroutine until an "awaitable" object (such as another coroutine, a Task, or a Future) completes. This error occurs when the object that await is applied to is not an awaitable but is instead the None singleton.

A critical aspect of this failure is the deceptive nature of the TypeError itself. The message immediately draws attention to the await keyword, suggesting a potential misuse of asynchronous syntax. The initial debugging efforts, as documented in the logs, correctly followed this line of inquiry, investigating whether an async with statement was being improperly awaited or if a synchronous function was being called without being wrapped in run\_in\_executor.1

However, this focus on the await syntax is a red herring. The syntax itself is correct; the problem lies with the operand. The investigation correctly pivoted from analyzing *how* the await was used to questioning *what* was being awaited. The central question is not "Why is this await expression invalid?" but rather "Why is self.llm\_service evaluating to None at this point in the execution?" This shift in perspective proved essential for moving beyond the symptoms and uncovering the true root cause of the failure.

## **Hypothesis-Driven Debugging: A Chronological Investigation**

The path to identifying the root cause was a systematic process of forming hypotheses, designing tests to validate them, and using the resulting evidence to refine the next line of inquiry. This process, documented across DEBUGGING\_LOG.txt and LLM\_CHAT.txt, systematically eliminated potential causes and converged on the definitive source of the error.1

The following table summarizes the key phases of this investigation.

| Hypothesis | Action/Tool Used | Key Observation/Evidence | Outcome/Next Step |  |
| :---- | :---- | :---- | :---- | :---- |
| **1\. Incorrect async with usage.** The performance\_optimizer context manager is being awaited directly. | Code review of llm\_service.py; execution of test\_context\_manager\_issue.py. | The context manager is correctly used with async with. The test script confirms that awaiting it directly raises a different error: \_AsyncGeneratorContextManager can't be used in 'await' expression.1 | **Hypothesis Disproven.** The issue is not with the context manager syntax itself. |  |
| **2\. \_generate\_with\_timeout returns None.** The method has a code path (e.g., in an exception handler) that fails to return a value. | Code review of \_generate\_with\_timeout in llm\_service.py. | The method was found to have a missing raise or return in some exception blocks. This was fixed, but the primary error persisted.1 | **Hypothesis Partially Valid (but not the root cause).** The method was improved, but the core bug remained. |  |
| **3\. loop.run\_in\_executor is used incorrectly.** The call to run the synchronous \_make\_llm\_api\_call in a thread is misconfigured. | Code review and modification of the run\_in\_executor call in llm\_service.py. | The call was initially loop.run\_in\_executor(None,...). This was changed to loop.run\_in\_executor(self.executor,...) for consistency, but the error persisted.1 | **Hypothesis Disproven.** The executor usage was not the root cause. |  |
| **4\. The llm\_service instance itself is None.** The object being awaited (self.llm\_service) is None when the generate\_content method is called. | Traceback analysis from app-log.txt.1 Code review of | agent\_base.py. | The traceback clearly shows the TypeError occurs on await self.llm\_service.generate\_content(...). This confirms the object self.llm\_service is None. | **Hypothesis Confirmed.** The investigation must now focus on why self.llm\_service is None. |
| **5\. The get\_llm\_service() singleton getter fails silently.** The service's constructor fails, and the getter function returns None instead of propagating the error. | Code review of get\_llm\_service() and EnhancedLLMService.\_\_init\_\_ in llm\_service.py.1 | The \_\_init\_\_ method raises a ValueError if no API key is found. The get\_llm\_service function uses a broad try...except that catches this ValueError but fails to re-raise it, allowing the global \_llm\_service\_instance to remain None. | **Root Cause Identified.** The combination of a failing constructor and a fragile singleton getter is the source of the None value. |  |

### **Phase 1: Investigating the Asynchronous Call Site**

The initial phase of the investigation correctly focused on the site of the await expression. Given the complexity of modern asynchronous Python, common pitfalls include misusing context managers or improperly mixing synchronous and asynchronous code. The performance\_optimizer.optimized\_execution is an async context manager and was a primary suspect. However, a review of the code and the execution of a dedicated test script, test\_context\_manager\_issue.py, confirmed it was being used correctly with the async with statement.1 Similarly, the investigation explored the use of

loop.run\_in\_executor, a function for running blocking I/O in a separate thread to prevent it from blocking the main asyncio event loop.2 While minor improvements were made for consistency, this was also ruled out as the root cause.

### **Phase 2: Tracing the None Value**

With the syntax of the await call exonerated, the investigation correctly pivoted to analyzing the operand: self.llm\_service. The traceback from app-log.txt was irrefutable evidence that the object held by this attribute was None at the time of the call.1 This shifted the focus from the

asyncio runtime to the application's dependency management and object lifecycle.

The call stack was traced backwards: the ParserAgent receives self.llm\_service during its initialization. This service is provided by the get\_llm\_service() factory function in src/services/llm\_service.py. This function employs a simple singleton pattern, using a global variable \_llm\_service\_instance to store the service object after its first creation. This pattern, while common, is inherently fragile. If the initial instantiation, EnhancedLLMService(), fails for any reason, the global variable is never assigned. Subsequent calls to get\_llm\_service() will then return the initial value of the global variable, which is None. This pattern effectively hides the original initialization failure and allows a null dependency to be injected deep into the application, leading to a deferred and obscure crash far from the original point of error. This architectural choice is a significant contributing factor to the bug's difficulty.

### **Phase 3: Uncovering the Initialization Failure**

The final phase of the investigation focused on the EnhancedLLMService.\_\_init\_\_ method to determine why it might be failing.1 The code revealed a clear failure point: the constructor checks for the presence of a

GEMINI\_API\_KEY from the application's settings. If no primary or fallback key is found, it raises a ValueError.1

Connecting this to the behavior of the singleton getter in Phase 2 completes the causal chain. The get\_llm\_service function's try...except block was catching this specific ValueError, logging a warning (which may be missed), and then returning None. This confirmed that a missing or misconfigured API key was the ultimate trigger for the entire failure sequence.

## **Root Cause Analysis: A Confluence of Failures**

The TypeError that halts the aicvgen workflow is not the result of a single mistake but rather a confluence of a proximate configuration error and several underlying architectural deficiencies. These factors combined to create a failure mode that was both critical and difficult to diagnose.

### **4.1. Proximate Cause: Silent Initialization Failure in EnhancedLLMService**

The direct and immediate cause of the bug is a silent failure during the instantiation of the EnhancedLLMService. The failure occurs within the get\_llm\_service factory function, which is designed to act as a singleton provider for the LLM service instance.

The precise failure chain is as follows:

1. The application starts, and at some point, a component requests the LLM service for the first time by calling get\_llm\_service().  
2. Inside get\_llm\_service, the global variable \_llm\_service\_instance is checked and found to be None.  
3. The code proceeds to instantiate the service: \_llm\_service\_instance \= EnhancedLLMService().  
4. The EnhancedLLMService.\_\_init\_\_ constructor is executed. It attempts to load the GEMINI\_API\_KEY from the application settings. Due to a missing .env file or a misconfiguration, no API key is found.  
5. The constructor correctly raises a ValueError with the message: "No Gemini API key found...".  
6. **This is the critical failure point.** The get\_llm\_service function wraps the instantiation in a broad try...except Exception block. This block catches the ValueError, logs a warning, and then implicitly returns None because there is no return statement in the except block.  
7. The \_llm\_service\_instance global variable is never assigned and remains None.  
8. This None value is returned to the caller, which injects it as a dependency into an agent (e.g., ParserAgent's self.llm\_service attribute).  
9. Later, the agent attempts to use the service by calling await self.llm\_service.generate\_content(...).  
10. This expression evaluates to await None.generate\_content(...), which triggers the fatal TypeError, as None is not an awaitable object and has no generate\_content method.

### **4.2. Contributing Factor 1: Inconsistent Asynchronous Patterns**

While not the direct cause of this specific bug, the codebase exhibits inconsistencies in its use of asyncio that point to a systemic risk. The pylint error report flags a critical E1142: 'await' should be used within an async function (await-outside-async) in src/agents/cv\_analyzer\_agent.py.1 This indicates that fundamental

asyncio syntax rules are not being universally followed, increasing the likelihood of other, more subtle asynchronous bugs.

Furthermore, the EnhancedLLMService itself must contend with the complexity of wrapping a synchronous library call (google-generativeai.GenerativeModel.generate\_content) within an asynchronous application.4 The correct pattern for this,

await loop.run\_in\_executor(...), is non-trivial and requires a solid understanding of the asyncio event loop and thread pool executors.2 While the implementation in

llm\_service.py is ultimately correct, the general complexity of mixing synchronous and asynchronous code throughout a large application is a significant source of technical debt and risk. This complexity can easily lead to issues like unintentionally blocking the event loop, which degrades performance, or creating race conditions that are notoriously difficult to debug.

### **4.3. Contributing Factor 2: Brittle Service Layer Architecture**

This bug is a textbook example of the architectural weaknesses identified in the "Codebase Analysis & Technical Debt Audit".1 The service layer, as designed, lacks robustness and violates the "fail-fast" principle.

A resilient system should have detected the missing API key at startup and crashed immediately with a clear, fatal ConfigurationError. Instead, the combination of the fragile singleton getter and the overly broad exception handling allowed the application to continue running in a broken state, hiding the root cause. This design choice defers the failure to a later, seemingly unrelated part of the code, dramatically increasing the difficulty of debugging.

Moreover, the application logs show the system attempting to retry the failed LLM call multiple times.1 This is another architectural flaw highlighted in the audit: the retry logic is attempting to recover from a fatal, non-transient configuration error. Retries are appropriate for temporary network issues or rate-limiting, not for a missing API key. This behavior wastes system resources, adds significant delays to the user-facing error, and further obscures the true nature of the problem.

## **The Implemented Solution and Verification**

The resolution of BUG-aicvgen-002 required targeted code-level fixes within src/services/llm\_service.py to enforce a fail-fast policy and provide clear error messaging. The verification process involved a combination of unit and integration tests to confirm the fix and validate the new, more robust behavior.

### **5.1. Code-Level Fixes for src/services/llm\_service.py**

The following table details the precise modifications made to the service layer to correct the silent failure behavior.

| Method | Change Type | Code Diff |
| :---- | :---- | :---- |
| \_\_init\_\_ | Modification | diff\<br\>--- a/src/services/llm\_service.py\<br\>+++ b/src/services/llm\_service.py\<br\>@@ \-123,10 \+123,19 @@\<br\> self.fallback\_api\_key \= self.settings.llm.gemini\_api\_key\_fallback\<br\> \<br\> \# Prioritize user-provided key, then primary, then fallback\<br\>- if self.user\_api\_key:\<br\>- api\_key \= self.user\_api\_key\<br\>- self.using\_user\_key \= True\<br\>- elif self.primary\_api\_key:\<br\>- api\_key \= self.primary\_api\_key\<br\>- self.using\_user\_key \= False\<br\>- elif self.fallback\_api\_key:\<br\>- api\_key \= self.fallback\_api\_key\<br\>- self.using\_user\_key \= False\<br\>- else:\<br\>- raise ValueError(\<br\>- "No Gemini API key found. Please provide your API key or "\<br\>- "set GEMINI\_API\_KEY environment variable."\<br\>- )\<br\>+ api\_key \= self.user\_api\_key or self.primary\_api\_key or self.fallback\_api\_key\<br\>+ if not api\_key:\<br\>+ \# Fail fast with a clear, actionable error message.\<br\>+ raise ConfigurationError(\<br\>+ "CRITICAL: Gemini API key is not configured. " \<br\>+ "Please set the GEMINI\_API\_KEY in your.env file or provide it in the UI. " \<br\>+ "Application cannot start without a valid API key."\<br\>+ )\<br\>+\<br\>+ self.using\_user\_key \= bool(self.user\_api\_key)\<br\> self.current\_api\_key \= api\_key\<br\> \<br\> \# Initialize the model\<br\> |
| get\_llm\_service | Modification | diff\<br\>--- a/src/services/llm\_service.py\<br\>+++ b/src/services/llm\_service.py\<br\>@@ \-80,14 \+80,12 @@\<br\> def get\_llm\_service(user\_api\_key: Optional\[str\] \= None) \-\> Optional:\<br\> """Get the global LLM service instance (singleton)."""\<br\> global \_llm\_service\_instance\<br\> if \_llm\_service\_instance is None:\<br\>- try:\<br\>- \_llm\_service\_instance \= EnhancedLLMService(user\_api\_key=user\_api\_key)\<br\>- logger.info("LLM service singleton created.")\<br\>- except Exception as e:\<br\>- logger.error(\<br\>- f"Failed to initialize EnhancedLLMService: {e}",\<br\>- exc\_info=True,\<br\>- error\_type=type(e).\_\_name\_\_,\<br\>- )\<br\>- return None\<br\>+ \# The constructor will now raise a ConfigurationError if setup fails,\<br\>+ \# ensuring a fail-fast behavior. No try/except block is needed here\<br\>+ \# as this critical error should halt the application.\<br\>+ \_llm\_service\_instance \= EnhancedLLMService(user\_api\_key=user\_api\_key)\<br\>+ logger.info("LLM service singleton created.")\<br\> return \_llm\_service\_instance\<br\> |

**Fix 1: Robust get\_llm\_service Getter:** The primary fix was to remove the overly broad try...except Exception block from the get\_llm\_service function. This ensures that any exception raised by the EnhancedLLMService constructor, such as the ConfigurationError for a missing API key, is no longer caught and silenced. Instead, the exception will propagate up the call stack, causing the application to fail immediately at startup, which is the correct behavior for a critical configuration issue.

**Fix 2: Explicit Validation in \_\_init\_\_:** The ValueError in the EnhancedLLMService constructor was replaced with a more specific ConfigurationError. The error message was enhanced to be more explicit and actionable, clearly stating that the API key is missing and instructing the user on how to resolve the issue (e.g., by setting the GEMINI\_API\_KEY in the .env file).

### **5.2. Verification Strategy**

The efficacy of these fixes was validated through a multi-layered testing strategy:

* **Unit Testing:** The existing test suite was augmented with new tests, such as test\_api\_key.py and test\_llm\_debug.py, to specifically target the EnhancedLLMService initialization.1 These tests now assert that a  
  ConfigurationError is raised when the service is instantiated without a valid API key.  
* **Integration Testing:** The full integration test suite, including tests/integration/test\_agent\_workflow\_integration.py, was executed under a deliberately misconfigured environment (i.e., with the GEMINI\_API\_KEY removed from the test environment).1 The successful outcome is no longer a completed workflow with errors, but an immediate and clear  
  ConfigurationError during the test setup phase. This confirms the fail-fast principle has been correctly implemented.  
* **Manual Verification:** Manually running the application via streamlit run app.py without a valid API key now results in an immediate crash with a clear traceback in the terminal, rather than launching a non-functional UI. This provides definitive confirmation that the silent failure mode has been eliminated.

## **Strategic Recommendations for Long-Term Stability**

Fixing BUG-aicvgen-002 stabilizes the application against one critical failure mode. However, to build a truly robust and maintainable system, the underlying architectural issues that allowed this bug to occur must be addressed. The following recommendations, drawn from the "Codebase Analysis & Technical Debt Audit," should be prioritized to improve the long-term health of the aicvgen project.1

### **Recommendation 1: Centralize and Simplify Resilience Patterns**

The audit identified duplicated and overlapping resilience logic (retries, rate limiting) in both llm\_service.py and item\_processor.py.1 This creates complex, unpredictable behavior, as demonstrated by the system attempting to retry a fatal configuration error.

**Action:** Refactor the codebase to establish the EnhancedLLMService as the single, authoritative source for all resilience patterns related to the Gemini API. This includes retries with exponential backoff, rate limiting, and circuit breaking. Higher-level components like agents or the ItemProcessor should not implement their own retry loops for LLM calls. They should simply call the EnhancedLLMService, which will return a final success or failure result after its internal resilience logic has been exhausted. This simplifies the control flow, makes failure behavior predictable, and eliminates the risk of cascading retry storms.

### **Recommendation 2: Implement a Dependency Injection Framework**

The fragile singleton pattern (get\_llm\_service) was a key contributor to this bug, as it hid the initial ValueError and propagated a None object as a valid dependency.

**Action:** Integrate a formal dependency injection (DI) framework (e.g., dependency-injector or a custom, lightweight container). The DI container would be responsible for the application's object graph. At startup, it would instantiate and configure all services. If EnhancedLLMService requires an API key, the container would verify its presence before even attempting to create the service instance. This moves configuration validation to the earliest possible point in the application lifecycle and ensures that no component can ever be instantiated with a missing or null dependency. This replaces the fragile, manual singleton pattern with a robust, explicit, and self-validating system for managing dependencies.

### **Recommendation 3: Enforce Stricter CI Quality Gates**

The pylint report revealed a critical await-outside-async error in cv\_analyzer\_agent.py, indicating that fundamental code quality issues can currently be merged into the codebase.1

**Action:** Configure the project's Continuous Integration (CI) pipeline (e.g., in GitHub Actions) to execute pylint src \--errors-only as a required status check. The build should be configured to fail if this command produces any output. This establishes an automated quality gate that prevents code with Error (E) or Fatal (F) level Pylint violations from ever being merged into the main branch. This simple change would have caught the await-outside-async error and will prevent a wide range of similar high-severity bugs in the future.

### **Recommendation 4: Address Agent Data Contract Breaches (Task 2.1)**

The "Codebase Analysis" and TASK\_BLUEPRINT.txt both identify a critical data flow problem: agents are not returning data in the format expected by the AgentState Pydantic model.1 For example, an agent might return data under the key

cv\_analysis\_results when the orchestrator expects quality\_check\_results. This leads to data being lost between workflow nodes and is the direct cause of a different class of NoneType errors when a subsequent node tries to access data that was never correctly populated.

**Action:** Prioritize the immediate execution of Task 2.1: Align Agent run\_as\_node Outputs with AgentState. This task involves refactoring every agent's run\_as\_node method to ensure its return dictionary's keys are a perfect match for the attribute names in the AgentState model. This is the next most critical stabilization task and will prevent a wave of data-flow-related NoneType errors that will likely surface now that the service initialization bug is resolved.

## **Conclusion**

The investigation into BUG-aicvgen-002 has revealed that the TypeError: object NoneType can't be used in 'await' expression was not a simple coding error but the result of a "perfect storm" of interconnected issues. The proximate cause was a silent failure in the EnhancedLLMService constructor due to a missing API key. This failure was then obscured by a fragile singleton pattern that propagated a None dependency, and the resulting crash was made more confusing by an application-wide retry mechanism attempting to recover from an unrecoverable error.

This incident serves as a crucial case study in the development of complex, multi-agent AI systems. It underscores the absolute necessity of fail-fast design principles; critical configuration errors must cause an immediate and loud failure at startup, not a deferred and silent one at runtime. It also highlights the importance of creating robust, well-defined service layers and establishing clear architectural ownership for cross-cutting concerns like error handling, dependency management, and resilience.

By implementing the immediate code-level fixes and committing to the strategic architectural recommendations outlined in this report, the anasakhomach-aicvgen project can move beyond simple bug-fixing and begin the necessary work of paying down its technical debt. This will establish the foundational stability required to build a reliable, maintainable, and scalable AI application.

#### **Works cited**

1. TASK\_BLUEPRINT.txt  
2. Python Asyncio Part 5 – Mixing Synchronous and Asynchronous Code | cloudfit-public-docs, accessed on June 19, 2025, [https://bbc.github.io/cloudfit-public-docs/asyncio/asyncio-part-5.html](https://bbc.github.io/cloudfit-public-docs/asyncio/asyncio-part-5.html)  
3. How Can I Run Synchronous Function From Asynchronous Function? : r/learnpython \- Reddit, accessed on June 19, 2025, [https://www.reddit.com/r/learnpython/comments/w1c43w/how\_can\_i\_run\_synchronous\_function\_from/](https://www.reddit.com/r/learnpython/comments/w1c43w/how_can_i_run_synchronous_function_from/)  
4. TypeError: object GenerateContentResponse can't be used in 'await' expression with model.generate\_content\_async() in Google Colab (Python 3.11) · Issue \#732 \- GitHub, accessed on June 19, 2025, [https://github.com/google-gemini/deprecated-generative-ai-python/issues/732](https://github.com/google-gemini/deprecated-generative-ai-python/issues/732)  
5. Generative AI on Vertex AI | Google Cloud, accessed on June 19, 2025, [https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/1.63.0/services](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/1.63.0/services)