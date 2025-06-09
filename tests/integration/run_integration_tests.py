"""Integration Test Runner for Phase 3 Testing.

Executes all integration tests and provides comprehensive reporting
for Orchestrator ↔ Agents, Agent ↔ Services, and State Persistence scenarios.
"""

import unittest
import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from io import StringIO

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import test modules
from test_orchestrator_agent_integration import TestOrchestratorAgentIntegration
from test_agent_services_integration import TestAgentServicesIntegration
from test_state_persistence_integration import TestStatePersistenceIntegration


class IntegrationTestRunner:
    """Comprehensive integration test runner with detailed reporting."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize test runner.
        
        Args:
            output_dir: Directory to save test reports. Defaults to current directory.
        """
        self.output_dir = output_dir or os.getcwd()
        self.test_results = {
            "orchestrator_agent": {},
            "agent_services": {},
            "state_persistence": {},
            "summary": {}
        }
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all integration tests and generate comprehensive report.
        
        Args:
            verbose: Whether to print detailed output during test execution.
            
        Returns:
            Dictionary containing test results and summary.
        """
        self.start_time = datetime.now()
        
        if verbose:
            print("\n" + "="*80)
            print("PHASE 3 INTEGRATION TESTS - EXECUTION STARTED")
            print("="*80)
            print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("\n")
        
        # Run test suites
        test_suites = [
            ("orchestrator_agent", TestOrchestratorAgentIntegration, "Orchestrator ↔ Agents Integration"),
            ("agent_services", TestAgentServicesIntegration, "Agent ↔ Services Integration"),
            ("state_persistence", TestStatePersistenceIntegration, "State Persistence Integration")
        ]
        
        for suite_key, test_class, description in test_suites:
            if verbose:
                print(f"\n{'='*60}")
                print(f"RUNNING: {description}")
                print(f"{'='*60}")
            
            suite_results = self._run_test_suite(test_class, verbose)
            self.test_results[suite_key] = suite_results
            
            if verbose:
                self._print_suite_summary(description, suite_results)
        
        self.end_time = datetime.now()
        
        # Generate summary
        self._generate_summary()
        
        if verbose:
            self._print_final_summary()
        
        # Save detailed report
        self._save_detailed_report()
        
        return self.test_results
    
    def _run_test_suite(self, test_class, verbose: bool = True) -> Dict[str, Any]:
        """Run a specific test suite and capture results.
        
        Args:
            test_class: Test class to execute.
            verbose: Whether to print test output.
            
        Returns:
            Dictionary containing suite results.
        """
        suite_start = time.time()
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        # Capture output
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2 if verbose else 1,
            buffer=True
        )
        
        # Run tests
        result = runner.run(suite)
        
        suite_end = time.time()
        
        # Process results
        suite_results = {
            "total_tests": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors),
            "failed": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
            "execution_time": suite_end - suite_start,
            "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
            "failures": self._format_test_failures(result.failures),
            "errors": self._format_test_errors(result.errors),
            "output": stream.getvalue()
        }
        
        if verbose and (result.failures or result.errors):
            print("\nFAILURES AND ERRORS:")
            print("-" * 40)
            for failure in result.failures:
                print(f"FAIL: {failure[0]}")
                print(failure[1])
                print("-" * 40)
            
            for error in result.errors:
                print(f"ERROR: {error[0]}")
                print(error[1])
                print("-" * 40)
        
        return suite_results
    
    def _format_test_failures(self, failures: List) -> List[Dict[str, str]]:
        """Format test failures for reporting."""
        return [
            {
                "test_name": str(failure[0]),
                "failure_message": failure[1]
            }
            for failure in failures
        ]
    
    def _format_test_errors(self, errors: List) -> List[Dict[str, str]]:
        """Format test errors for reporting."""
        return [
            {
                "test_name": str(error[0]),
                "error_message": error[1]
            }
            for error in errors
        ]
    
    def _generate_summary(self):
        """Generate overall test execution summary."""
        total_tests = sum(suite["total_tests"] for suite in self.test_results.values() if isinstance(suite, dict) and "total_tests" in suite)
        total_passed = sum(suite["passed"] for suite in self.test_results.values() if isinstance(suite, dict) and "passed" in suite)
        total_failed = sum(suite["failed"] for suite in self.test_results.values() if isinstance(suite, dict) and "failed" in suite)
        total_errors = sum(suite["errors"] for suite in self.test_results.values() if isinstance(suite, dict) and "errors" in suite)
        total_execution_time = sum(suite["execution_time"] for suite in self.test_results.values() if isinstance(suite, dict) and "execution_time" in suite)
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "overall_success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "total_execution_time": total_execution_time,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
        }
    
    def _print_suite_summary(self, description: str, results: Dict[str, Any]):
        """Print summary for a test suite."""
        print(f"\n{description} Results:")
        print(f"  Tests Run: {results['total_tests']}")
        print(f"  Passed: {results['passed']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Errors: {results['errors']}")
        print(f"  Success Rate: {results['success_rate']:.1f}%")
        print(f"  Execution Time: {results['execution_time']:.2f}s")
    
    def _print_final_summary(self):
        """Print final test execution summary."""
        summary = self.test_results["summary"]
        
        print("\n" + "="*80)
        print("PHASE 3 INTEGRATION TESTS - FINAL SUMMARY")
        print("="*80)
        
        print(f"\nExecution Period:")
        print(f"  Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {summary['duration']:.2f}s")
        
        print(f"\nOverall Results:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['total_passed']}")
        print(f"  Failed: {summary['total_failed']}")
        print(f"  Errors: {summary['total_errors']}")
        print(f"  Success Rate: {summary['overall_success_rate']:.1f}%")
        
        # Status indicator
        if summary['total_failed'] == 0 and summary['total_errors'] == 0:
            status = "✅ ALL TESTS PASSED"
        elif summary['overall_success_rate'] >= 80:
            status = "⚠️  MOSTLY SUCCESSFUL (some issues)"
        else:
            status = "❌ SIGNIFICANT ISSUES DETECTED"
        
        print(f"\nStatus: {status}")
        
        # Detailed breakdown by suite
        print(f"\nDetailed Breakdown:")
        suite_names = {
            "orchestrator_agent": "Orchestrator ↔ Agents",
            "agent_services": "Agent ↔ Services", 
            "state_persistence": "State Persistence"
        }
        
        for suite_key, suite_name in suite_names.items():
            if suite_key in self.test_results:
                suite = self.test_results[suite_key]
                status_icon = "✅" if suite['failed'] == 0 and suite['errors'] == 0 else "❌"
                print(f"  {status_icon} {suite_name}: {suite['passed']}/{suite['total_tests']} ({suite['success_rate']:.1f}%)")
        
        print("\n" + "="*80)
    
    def _save_detailed_report(self):
        """Save detailed test report to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"integration_test_report_{timestamp}.json")
        
        # Prepare report data
        report_data = {
            "test_execution": {
                "timestamp": timestamp,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": self.test_results["summary"]["duration"]
            },
            "results": self.test_results,
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": os.getcwd()
            }
        }
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nDetailed report saved to: {report_file}")
            
        except Exception as e:
            print(f"\nWarning: Could not save detailed report: {e}")
    
    def run_specific_suite(self, suite_name: str, verbose: bool = True) -> Dict[str, Any]:
        """Run a specific test suite.
        
        Args:
            suite_name: Name of the suite ('orchestrator_agent', 'agent_services', 'state_persistence')
            verbose: Whether to print detailed output.
            
        Returns:
            Dictionary containing suite results.
        """
        suite_mapping = {
            "orchestrator_agent": (TestOrchestratorAgentIntegration, "Orchestrator ↔ Agents Integration"),
            "agent_services": (TestAgentServicesIntegration, "Agent ↔ Services Integration"),
            "state_persistence": (TestStatePersistenceIntegration, "State Persistence Integration")
        }
        
        if suite_name not in suite_mapping:
            raise ValueError(f"Unknown suite: {suite_name}. Available: {list(suite_mapping.keys())}")
        
        test_class, description = suite_mapping[suite_name]
        
        if verbose:
            print(f"\nRunning {description}...")
        
        results = self._run_test_suite(test_class, verbose)
        
        if verbose:
            self._print_suite_summary(description, results)
        
        return results


def main():
    """Main entry point for integration test execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phase 3 Integration Tests")
    parser.add_argument(
        "--suite",
        choices=["orchestrator_agent", "agent_services", "state_persistence", "all"],
        default="all",
        help="Specific test suite to run (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save test reports (default: current directory)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = IntegrationTestRunner(output_dir=args.output_dir)
    
    try:
        if args.suite == "all":
            results = runner.run_all_tests(verbose=not args.quiet)
        else:
            results = runner.run_specific_suite(args.suite, verbose=not args.quiet)
        
        # Exit with appropriate code
        if args.suite == "all":
            exit_code = 0 if results["summary"]["total_failed"] == 0 and results["summary"]["total_errors"] == 0 else 1
        else:
            exit_code = 0 if results["failed"] == 0 and results["errors"] == 0 else 1
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\nError running integration tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()