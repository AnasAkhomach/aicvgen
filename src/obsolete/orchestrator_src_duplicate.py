"""Enhanced orchestrator for individual item processing workflow.

This module provides the main orchestration logic for processing CV items
individually to mitigate LLM rate limits while maintaining workflow integrity."""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict

from ..config.logging_config import get_structured_logger
from ..config.settings import get_config
from ..models.data_models import (
    CVGenerationState, WorkflowStage, ProcessingStatus, ContentType,
    ContentItem, ExperienceItem, ProjectItem, QualificationItem,
    JobDescriptionData, ProcessingQueue
)
from ..services.item_processor import ItemProcessor
from ..services.rate_limiter import get_rate_limiter
from ..services.progress_tracker import get_progress_tracker, ProgressTracker
from ..services.error_recovery import get_error_recovery_service, ErrorRecoveryService
from ..services.session_manager import get_session_manager, SessionManager, SessionStatus


class EnhancedOrchestrator:
    """Enhanced orchestrator with individual item processing capabilities."""
    
    def __init__(
        self,
        llm_client=None,
        progress_tracker: Optional[ProgressTracker] = None,
        error_recovery: Optional[ErrorRecoveryService] = None,
        session_manager: Optional[SessionManager] = None,
        progress_callback: Optional[Callable] = None
    ):
        self.llm_client = llm_client
        self.item_processor = ItemProcessor(llm_client)
        self.rate_limiter = get_rate_limiter()
        self.logger = get_structured_logger("enhanced_orchestrator")
        self.settings = get_config()
        
        # Enhanced services
        self.progress_tracker = progress_tracker or get_progress_tracker()
        self.error_recovery = error_recovery or get_error_recovery_service()
        self.session_manager = session_manager or get_session_manager()
        
        self.progress_callback = progress_callback
        
        # Workflow state
        self.current_state: Optional[CVGenerationState] = None
        self.is_processing = False
        self.processing_task: Optional[asyncio.Task] = None
    
    async def start_cv_generation(
        self,
        job_description: str,
        user_data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> CVGenerationState:
        """Start the CV generation workflow.
        
        Args:
            job_description: Raw job description text
            user_data: User's CV data (experiences, projects, etc.)
            session_id: Optional session ID for tracking
            
        Returns:
            CVGenerationState: Initial workflow state
        """
        
        self.logger.info("Starting CV generation workflow", session_id=session_id)
        
        try:
            # Get or create session
            if not session_id:
                session_id = f"session_{int(time.time())}"
            
            session_info = self.session_manager.get_session(session_id)
            if not session_info:
                session_id = self.session_manager.create_session(
                    metadata={
                        "job_description_length": len(job_description),
                        "user_data_keys": list(user_data.keys())
                    }
                )
            
            # Initialize workflow state
            self.current_state = CVGenerationState(session_id=session_id)
            
            # Start progress tracking
            self.progress_tracker.start_tracking(session_id, self.current_state)
            
            # Parse job description
            await self._parse_job_description(job_description)
            
            # Setup processing queues
            await self._setup_processing_queues(user_data)
            
            # Update stage
            self.current_state.update_stage(WorkflowStage.QUALIFICATION_GENERATION)
            
            # Update session state
            self.session_manager.update_session_state(session_id, self.current_state)
            
            self.logger.info(
                "CV generation workflow initialized",
                session_id=self.current_state.session_id,
                total_items=(
                    self.current_state.qualification_queue.total_items +
                    self.current_state.experience_queue.total_items +
                    self.current_state.project_queue.total_items
                )
            )
            
            return self.current_state
            
        except Exception as e:
            self.logger.error(
                f"CV generation failed: {e}",
                session_id=session_id
            )
            
            if session_id:
                self.session_manager.fail_session(session_id, str(e))
                self.progress_tracker.record_error(session_id, str(e))
            
            raise
    
    async def process_next_batch(self, batch_size: int = 3) -> Dict[str, Any]:
        """Process the next batch of items.
        
        Args:
            batch_size: Number of items to process in parallel
            
        Returns:
            Dict with processing results and progress
        """
        
        if not self.current_state:
            raise ValueError("No active CV generation session")
        
        if self.is_processing:
            return {
                "status": "already_processing",
                "message": "Processing is already in progress"
            }
        
        self.is_processing = True
        start_time = time.time()
        
        try:
            # Get items ready for processing
            items_to_process = self._get_next_items_for_processing(batch_size)
            
            if not items_to_process:
                # Check if we need to move to next stage or complete
                return await self._handle_stage_completion()
            
            # Process items in parallel
            results = await self._process_items_batch(items_to_process)
            
            # Update progress
            processing_time = time.time() - start_time
            self.current_state.total_processing_time += processing_time
            
            # Call progress callback if provided
            if self.progress_callback:
                await self._notify_progress()
            
            return {
                "status": "batch_completed",
                "processed_items": len([r for r in results if r["success"]]),
                "failed_items": len([r for r in results if not r["success"]]),
                "processing_time": processing_time,
                "overall_progress": self.current_state.overall_progress,
                "next_batch_available": len(self._get_next_items_for_processing(batch_size)) > 0
            }
            
        finally:
            self.is_processing = False
    
    async def process_all_items(self, batch_size: int = 3, delay_between_batches: float = 1.0) -> Dict[str, Any]:
        """Process all items in the workflow.
        
        Args:
            batch_size: Number of items to process in parallel
            delay_between_batches: Delay between batches to respect rate limits
            
        Returns:
            Dict with final processing results
        """
        
        if not self.current_state:
            raise ValueError("No active CV generation session")
        
        self.logger.info(
            "Starting complete CV generation processing",
            session_id=self.current_state.session_id,
            batch_size=batch_size
        )
        
        total_start_time = time.time()
        batch_count = 0
        
        while True:
            batch_result = await self.process_next_batch(batch_size)
            batch_count += 1
            
            self.logger.info(
                f"Completed batch {batch_count}",
                batch_result=batch_result,
                session_id=self.current_state.session_id
            )
            
            # Check if processing is complete
            if batch_result["status"] in ["workflow_completed", "no_items_to_process"]:
                break
            
            # Check if there are more items to process
            if not batch_result.get("next_batch_available", False):
                # Try to advance to next stage
                stage_result = await self._handle_stage_completion()
                if stage_result["status"] == "workflow_completed":
                    break
            
            # Delay between batches to respect rate limits
            if delay_between_batches > 0:
                await asyncio.sleep(delay_between_batches)
        
        # Generate executive summary if not done
        if (self.current_state.current_stage != WorkflowStage.COMPLETED and 
            not self.current_state.executive_summary):
            await self._generate_executive_summary()
        
        # Final stage update
        self.current_state.update_stage(WorkflowStage.COMPLETED)
        
        total_time = time.time() - total_start_time
        
        final_result = {
            "status": "completed",
            "session_id": self.current_state.session_id,
            "total_processing_time": total_time,
            "batches_processed": batch_count,
            "overall_progress": self.current_state.overall_progress,
            "processor_stats": self.item_processor.get_processing_stats(),
            "final_state": self.current_state.is_complete
        }
        
        self.logger.info(
            "CV generation workflow completed",
            session_id=self.current_state.session_id,
            final_result=final_result
        )
        
        return final_result
    
    async def _parse_job_description(self, job_description: str):
        """Parse the job description to extract key information."""
        self.current_state.update_stage(WorkflowStage.JOB_PARSING)
        
        try:
            # Update progress tracking
            self.progress_tracker.update_progress(self.current_state.session_id, self.current_state)
            
            # For now, create a basic parsed structure
            # In a full implementation, this would use an LLM to extract structured data
            self.current_state.job_description = JobDescriptionData(
                raw_text=job_description,
                company_name="",  # Would be extracted
                position_title="",  # Would be extracted
                required_skills=[],  # Would be extracted
                preferred_skills=[],  # Would be extracted
                responsibilities=[],  # Would be extracted
                qualifications=[]  # Would be extracted
            )
            
            self.logger.info(
                "Job description parsed",
                session_id=self.current_state.session_id,
                text_length=len(job_description)
            )
            
        except Exception as e:
            await self.error_recovery.handle_error(
                e, "job_description", ContentType.QUALIFICATION,
                self.current_state.session_id, context={"stage": "job_parsing"}
            )
            raise
    
    async def _setup_processing_queues(self, user_data: Dict[str, Any]):
        """Setup processing queues with user data."""
        
        # Add qualifications (generate 10 as per requirements)
        base_qualifications = user_data.get('qualifications', [])
        if len(base_qualifications) < self.current_state.target_qualifications_count:
            # Pad with generic qualifications that will be tailored
            base_qualifications.extend([
                "Strong problem-solving and analytical skills",
                "Excellent communication and teamwork abilities",
                "Proficient in modern development practices",
                "Experience with agile methodologies",
                "Continuous learning and adaptation mindset"
            ][:self.current_state.target_qualifications_count - len(base_qualifications)])
        
        self.current_state.add_qualification_items(base_qualifications[:self.current_state.target_qualifications_count])
        
        # Add experiences
        experiences = user_data.get('experiences', [])
        self.current_state.add_experience_items(experiences)
        
        # Add projects
        projects = user_data.get('projects', [])
        self.current_state.add_project_items(projects)
        
        self.logger.info(
            "Processing queues setup",
            session_id=self.current_state.session_id,
            qualifications=len(base_qualifications),
            experiences=len(experiences),
            projects=len(projects)
        )
    
    def _get_next_items_for_processing(self, batch_size: int) -> List[ContentItem]:
        """Get the next items ready for processing."""
        items = []
        
        # Process based on current stage
        if self.current_state.current_stage == WorkflowStage.QUALIFICATION_GENERATION:
            # Get qualification items
            for _ in range(batch_size):
                item = self.current_state.qualification_queue.get_next_item()
                if item:
                    items.append(item)
                else:
                    break
        
        elif self.current_state.current_stage == WorkflowStage.EXPERIENCE_PROCESSING:
            # Get experience items
            for _ in range(batch_size):
                item = self.current_state.experience_queue.get_next_item()
                if item:
                    items.append(item)
                else:
                    break
        
        elif self.current_state.current_stage == WorkflowStage.PROJECT_PROCESSING:
            # Get project items
            for _ in range(batch_size):
                item = self.current_state.project_queue.get_next_item()
                if item:
                    items.append(item)
                else:
                    break
        
        return items
    
    async def _process_items_batch(self, items: List[ContentItem]) -> List[Dict[str, Any]]:
        """Process a batch of items in parallel."""
        
        job_context = self._build_job_context()
        
        # Create processing tasks
        tasks = []
        for item in items:
            task = asyncio.create_task(
                self._process_single_item_with_context(item, job_context)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            item = items[i]
            
            if isinstance(result, Exception):
                processed_results.append({
                    "item_id": item.metadata.item_id,
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append({
                    "item_id": item.metadata.item_id,
                    "success": result,
                    "status": item.metadata.status.value
                })
        
        return processed_results
    
    async def _process_single_item_with_context(
        self, 
        item: ContentItem, 
        job_context: Dict[str, Any]
    ) -> bool:
        """Process a single item with error handling."""
        try:
            return await self.item_processor.process_item(item, job_context)
        except Exception as e:
            self.logger.error(
                f"Error processing item {item.metadata.item_id}",
                error=str(e),
                item_type=item.content_type.value
            )
            return False
    
    def _build_job_context(self) -> Dict[str, Any]:
        """Build job context for item processing."""
        if not self.current_state.job_description:
            return {}
        
        return {
            "company_name": self.current_state.job_description.company_name,
            "position_title": self.current_state.job_description.position_title,
            "required_skills": self.current_state.job_description.required_skills,
            "preferred_skills": self.current_state.job_description.preferred_skills,
            "responsibilities": self.current_state.job_description.responsibilities,
            "qualifications": self.current_state.job_description.qualifications,
            "raw_text": self.current_state.job_description.raw_text
        }
    
    async def _handle_stage_completion(self) -> Dict[str, Any]:
        """Handle completion of current stage and advance to next."""
        
        current_stage = self.current_state.current_stage
        
        if current_stage == WorkflowStage.QUALIFICATION_GENERATION:
            # Check if all qualifications are processed
            if self.current_state.qualification_queue.completion_percentage == 100:
                self.current_state.update_stage(WorkflowStage.EXPERIENCE_PROCESSING)
                return {
                    "status": "stage_advanced",
                    "new_stage": WorkflowStage.EXPERIENCE_PROCESSING.value,
                    "message": "Advanced to experience processing"
                }
            else:
                return {
                    "status": "stage_incomplete",
                    "current_stage": current_stage.value,
                    "completion": self.current_state.qualification_queue.completion_percentage
                }
        
        elif current_stage == WorkflowStage.EXPERIENCE_PROCESSING:
            if self.current_state.experience_queue.completion_percentage == 100:
                self.current_state.update_stage(WorkflowStage.PROJECT_PROCESSING)
                return {
                    "status": "stage_advanced",
                    "new_stage": WorkflowStage.PROJECT_PROCESSING.value,
                    "message": "Advanced to project processing"
                }
            else:
                return {
                    "status": "stage_incomplete",
                    "current_stage": current_stage.value,
                    "completion": self.current_state.experience_queue.completion_percentage
                }
        
        elif current_stage == WorkflowStage.PROJECT_PROCESSING:
            if self.current_state.project_queue.completion_percentage == 100:
                self.current_state.update_stage(WorkflowStage.SUMMARY_GENERATION)
                await self._generate_executive_summary()
                return {
                    "status": "stage_advanced",
                    "new_stage": WorkflowStage.SUMMARY_GENERATION.value,
                    "message": "Advanced to summary generation"
                }
            else:
                return {
                    "status": "stage_incomplete",
                    "current_stage": current_stage.value,
                    "completion": self.current_state.project_queue.completion_percentage
                }
        
        elif current_stage == WorkflowStage.SUMMARY_GENERATION:
            if (self.current_state.executive_summary and 
                self.current_state.executive_summary.metadata.status == ProcessingStatus.COMPLETED):
                self.current_state.update_stage(WorkflowStage.COMPLETED)
                return {
                    "status": "workflow_completed",
                    "message": "CV generation workflow completed"
                }
        
        return {
            "status": "no_items_to_process",
            "current_stage": current_stage.value
        }
    
    async def _generate_executive_summary(self):
        """Generate the executive summary based on processed content."""
        
        # Collect processed content
        qualifications = [
            item.generated_content for item in self.current_state.qualification_queue.completed_items
            if item.generated_content
        ]
        
        experiences = [
            item.generated_content for item in self.current_state.experience_queue.completed_items
            if item.generated_content
        ]
        
        projects = [
            item.generated_content for item in self.current_state.project_queue.completed_items
            if item.generated_content
        ]
        
        # Create summary item
        summary_item = ContentItem(
            content_type=ContentType.EXECUTIVE_SUMMARY,
            original_content="Generate executive summary based on processed CV content"
        )
        
        # Build template context
        template_context = {
            "qualifications": qualifications,
            "experiences": experiences,
            "projects": projects
        }
        
        # Process the summary
        job_context = self._build_job_context()
        success = await self.item_processor.process_item(
            summary_item, job_context, template_context
        )
        
        if success:
            self.current_state.executive_summary = summary_item
            self.logger.info(
                "Executive summary generated",
                session_id=self.current_state.session_id
            )
        else:
            self.logger.error(
                "Failed to generate executive summary",
                session_id=self.current_state.session_id
            )
    
    async def _notify_progress(self):
        """Notify progress callback if provided."""
        if self.progress_callback:
            try:
                progress_data = {
                    "session_id": self.current_state.session_id,
                    "overall_progress": self.current_state.overall_progress,
                    "current_stage": self.current_state.current_stage.value,
                    "processor_stats": self.item_processor.get_processing_stats()
                }
                
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(progress_data)
                else:
                    self.progress_callback(progress_data)
                    
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
    
    def get_current_state(self) -> Optional[CVGenerationState]:
        """Get the current workflow state."""
        if self.current_state:
            return self.current_state
        return None
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress."""
        if not self.current_state:
            return {"status": "no_active_session"}
        
        # Get comprehensive progress data
        progress_data = self.progress_tracker.get_progress_summary(self.current_state.session_id)
        error_data = self.error_recovery.get_error_summary(self.current_state.session_id)
        session_info = self.session_manager.get_session(self.current_state.session_id)
        
        return {
            "session_id": self.current_state.session_id,
            "current_stage": self.current_state.current_stage.value,
            "overall_progress": self.current_state.overall_progress,
            "is_processing": self.is_processing,
            "processor_stats": self.item_processor.get_processing_stats(),
            "is_complete": self.current_state.is_complete,
            "progress_metrics": progress_data,
            "error_summary": error_data,
            "session_info": session_info.to_dict() if session_info else None
        }
    
    async def stop_processing(self):
        """Stop the current processing workflow."""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        self.is_processing = False
        self.logger.info("Processing stopped", session_id=self.current_state.session_id if self.current_state else None)