"""Content Aggregator for CV Generation System

This module provides the ContentAggregator class that collects individual content pieces
from different agents and assembles them into a complete ContentData structure.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ContentAggregator:
    """Aggregates individual content pieces from agents into a complete CV structure."""
    
    def __init__(self):
        """Initialize the content aggregator with content type mappings."""
        # Map content types to ContentData fields
        self.content_map = {
            'qualification': 'summary',
            'experience': 'experience_bullets',
            'skills': 'skills_section',
            'project': 'projects',
            'education': 'education',
            'certification': 'certifications',
            'language': 'languages'
        }
        
        # Initialize empty ContentData structure
        self.base_structure = {
            'summary': '',
            'experience_bullets': [],
            'skills_section': '',
            'projects': [],
            'education': [],
            'certifications': [],
            'languages': [],
            'contact_info': {}
        }
    
    def aggregate_results(self, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate task results into a complete ContentData structure.
        
        Args:
            task_results: List of task results from various agents
            
        Returns:
            Dictionary containing aggregated CV content in ContentData format
        """
        logger.info(f"Starting content aggregation for {len(task_results)} task results")
        
        # Start with base structure
        content_data = self.base_structure.copy()
        content_found = False
        
        for i, result in enumerate(task_results):
            try:
                agent_type = result.get('agent_type', 'unknown')
                logger.info(f"Processing result {i} from agent: {agent_type}")
                
                # Handle content_writer agent results
                if agent_type == 'content_writer':
                    content_found = self._process_content_writer_result(result, content_data) or content_found
                
                # Handle other agent types
                elif agent_type == 'content_optimization':
                    content_found = self._process_optimization_result(result, content_data) or content_found
                
                # Handle generic content structure
                else:
                    content_found = self._process_generic_result(result, content_data) or content_found
                    
            except Exception as e:
                logger.error(f"Error processing result {i}: {e}")
                continue
        
        if content_found:
            logger.info(f"Content aggregation successful. Fields populated: {[k for k, v in content_data.items() if v]}")
        else:
            logger.warning("No valid content found during aggregation")
        
        return content_data
    
    def _process_content_writer_result(self, result: Dict[str, Any], content_data: Dict[str, Any]) -> bool:
        """Process content writer agent results.
        
        Args:
            result: Task result from content writer agent
            content_data: Content data structure to populate
            
        Returns:
            True if content was successfully processed, False otherwise
        """
        try:
            # Extract content from AgentResult.output_data structure
            agent_content = result.get("content", {})
            
            if isinstance(agent_content, dict) and "content" in agent_content:
                # Extract the actual content from nested structure
                actual_content = agent_content["content"]
                content_type = agent_content.get("content_type", "unknown")
                
                logger.info(f"Found content writer content: type={content_type}, content_length={len(str(actual_content))}")
                
                # Map content type to appropriate field
                if content_type in self.content_map:
                    field = self.content_map[content_type]
                    
                    # Handle list fields vs string fields
                    if isinstance(content_data[field], list):
                        if isinstance(actual_content, list):
                            content_data[field].extend(actual_content)
                        else:
                            content_data[field].append(actual_content)
                    else:
                        content_data[field] = actual_content
                    
                    logger.info(f"Mapped {content_type} content to {field}")
                    return True
                else:
                    # Try to infer content type from content
                    return self._infer_and_assign_content(actual_content, content_data)
            
            # Handle direct string content (fallback)
            elif isinstance(agent_content, str):
                return self._infer_and_assign_content(agent_content, content_data)
                
        except Exception as e:
            logger.error(f"Error processing content writer result: {e}")
        
        return False
    
    def _process_optimization_result(self, result: Dict[str, Any], content_data: Dict[str, Any]) -> bool:
        """Process content optimization agent results.
        
        Args:
            result: Task result from optimization agent
            content_data: Content data structure to populate
            
        Returns:
            True if content was successfully processed, False otherwise
        """
        try:
            optimized_content = result.get("content", {}).get("optimized_content", {})
            
            if isinstance(optimized_content, dict):
                # Merge optimized content with existing structure
                for field in self.base_structure.keys():
                    if field in optimized_content and optimized_content[field]:
                        content_data[field] = optimized_content[field]
                        logger.info(f"Updated {field} from optimization agent")
                
                return any(optimized_content.values())
                
        except Exception as e:
            logger.error(f"Error processing optimization result: {e}")
        
        return False
    
    def _process_generic_result(self, result: Dict[str, Any], content_data: Dict[str, Any]) -> bool:
        """Process generic agent results.
        
        Args:
            result: Generic task result
            content_data: Content data structure to populate
            
        Returns:
            True if content was successfully processed, False otherwise
        """
        try:
            # Try to extract content from various possible structures
            content = None
            
            if "content" in result:
                content = result["content"]
            elif "result" in result and isinstance(result["result"], dict):
                if "content" in result["result"]:
                    content = result["result"]["content"]
            
            if content:
                if isinstance(content, dict):
                    # Try to merge structured content
                    for field in self.base_structure.keys():
                        if field in content and content[field]:
                            content_data[field] = content[field]
                    return True
                elif isinstance(content, str):
                    # Try to infer content type
                    return self._infer_and_assign_content(content, content_data)
                    
        except Exception as e:
            logger.error(f"Error processing generic result: {e}")
        
        return False
    
    def _infer_and_assign_content(self, content: str, content_data: Dict[str, Any]) -> bool:
        """Infer content type and assign to appropriate field.
        
        Args:
            content: String content to analyze and assign
            content_data: Content data structure to populate
            
        Returns:
            True if content was successfully assigned, False otherwise
        """
        try:
            content_lower = content.lower()
            
            # Simple heuristics to determine content type
            if any(keyword in content_lower for keyword in ['summary', 'profile', 'objective', 'about']):
                if not content_data['summary']:
                    content_data['summary'] = content
                    logger.info("Inferred content as summary")
                    return True
            
            elif any(keyword in content_lower for keyword in ['experience', 'work', 'employment', 'position']):
                content_data['experience_bullets'].append(content)
                logger.info("Inferred content as experience")
                return True
            
            elif any(keyword in content_lower for keyword in ['skill', 'technology', 'competency', 'expertise']):
                if not content_data['skills_section']:
                    content_data['skills_section'] = content
                    logger.info("Inferred content as skills")
                    return True
            
            elif any(keyword in content_lower for keyword in ['project', 'portfolio', 'development']):
                content_data['projects'].append(content)
                logger.info("Inferred content as project")
                return True
            
            else:
                # Default to summary if no specific type detected
                if not content_data['summary']:
                    content_data['summary'] = content
                    logger.info("Assigned content to summary as default")
                    return True
                else:
                    # Add to experience if summary is already filled
                    content_data['experience_bullets'].append(content)
                    logger.info("Assigned content to experience as fallback")
                    return True
                    
        except Exception as e:
            logger.error(f"Error inferring content type: {e}")
        
        return False
    
    def validate_content_data(self, content_data: Dict[str, Any]) -> bool:
        """Validate that the content data has meaningful content.
        
        Args:
            content_data: Content data to validate
            
        Returns:
            True if content data is valid, False otherwise
        """
        try:
            # Check if at least one major field has content
            major_fields = ['summary', 'experience_bullets', 'skills_section']
            
            for field in major_fields:
                if field in content_data and content_data[field]:
                    if isinstance(content_data[field], list) and len(content_data[field]) > 0:
                        return True
                    elif isinstance(content_data[field], str) and content_data[field].strip():
                        return True
            
            logger.warning("Content validation failed: no major fields populated")
            return False
            
        except Exception as e:
            logger.error(f"Error validating content data: {e}")
            return False