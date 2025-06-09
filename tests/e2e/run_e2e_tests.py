#!/usr/bin/env python3
"""End-to-End Test Runner for CV Tailoring Application.

This script provides a comprehensive test runner for executing E2E tests
with proper configuration, reporting, and result analysis.
"""

import sys
import os
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from datetime import datetime


@dataclass
class TestRunConfig:
    """Configuration for E2E test runs."""
    test_suite: str = "all"  # all, complete_cv, individual_item, error_recovery
    job_roles: List[str] = None  # None means all roles
    parallel_workers: int = 1
    timeout_seconds: int = 600
    verbose: bool = False
    generate_report: bool = True
    output_dir: Optional[str] = None
    mock_llm: bool = True
    performance_profiling: bool = False
    
    def __post_init__(self):
        if self.job_roles is None:
            self.job_roles = ["software_engineer", "ai_engineer", "data_scientist"]
        
        if self.output_dir is None:
            self.output_dir = str(project_root / "test_results" / "e2e" / datetime.now().strftime("%Y%m%d_%H%M%S"))


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None


@dataclass
class TestSuiteResults:
    """Complete test suite results."""
    config: TestRunConfig
    start_time: datetime
    end_time: datetime
    total_duration: float
    test_results: List[TestResult]
    summary: Dict[str, Any]
    performance_summary: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": asdict(self.config),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_duration": self.total_duration,
            "test_results": [asdict(result) for result in self.test_results],
            "summary": self.summary,
            "performance_summary": self.performance_summary
        }


class E2ETestRunner:
    """End-to-End test runner with comprehensive reporting."""
    
    def __init__(self, config: TestRunConfig):
        self.config = config
        self.results = TestSuiteResults(
            config=config,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,
            test_results=[],
            summary={}
        )
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_tests(self) -> TestSuiteResults:
        """Run the E2E test suite."""
        print(f"🚀 Starting E2E Test Suite: {self.config.test_suite}")
        print(f"📁 Output Directory: {self.config.output_dir}")
        print(f"🎯 Job Roles: {', '.join(self.config.job_roles)}")
        print(f"⚡ Parallel Workers: {self.config.parallel_workers}")
        print(f"🤖 Mock LLM: {self.config.mock_llm}")
        print("" + "="*60)
        
        self.results.start_time = datetime.now()
        
        try:
            # Build pytest arguments
            pytest_args = self._build_pytest_args()
            
            # Run tests
            exit_code = pytest.main(pytest_args)
            
            # Process results
            self._process_results(exit_code)
            
        except Exception as e:
            print(f"❌ Test execution failed: {str(e)}")
            self.results.summary["execution_error"] = str(e)
        
        finally:
            self.results.end_time = datetime.now()
            self.results.total_duration = (
                self.results.end_time - self.results.start_time
            ).total_seconds()
            
            # Generate reports
            if self.config.generate_report:
                self._generate_reports()
        
        return self.results
    
    def _build_pytest_args(self) -> List[str]:
        """Build pytest command line arguments."""
        args = []
        
        # Test directory
        test_dir = Path(__file__).parent
        
        # Select test files based on suite
        if self.config.test_suite == "all":
            args.append(str(test_dir))
        elif self.config.test_suite == "complete_cv":
            args.append(str(test_dir / "test_complete_cv_generation.py"))
        elif self.config.test_suite == "individual_item":
            args.append(str(test_dir / "test_individual_item_processing.py"))
        elif self.config.test_suite == "error_recovery":
            args.append(str(test_dir / "test_error_recovery.py"))
        else:
            raise ValueError(f"Unknown test suite: {self.config.test_suite}")
        
        # Markers
        args.extend(["-m", "e2e"])
        
        # Verbose output
        if self.config.verbose:
            args.append("-v")
        else:
            args.append("-q")
        
        # Parallel execution
        if self.config.parallel_workers > 1:
            args.extend(["-n", str(self.config.parallel_workers)])
        
        # Timeout
        args.extend(["--timeout", str(self.config.timeout_seconds)])
        
        # Output formats
        junit_file = Path(self.config.output_dir) / "junit_results.xml"
        args.extend(["--junit-xml", str(junit_file)])
        
        # HTML report
        html_file = Path(self.config.output_dir) / "test_report.html"
        args.extend(["--html", str(html_file), "--self-contained-html"])
        
        # Coverage (if enabled)
        if self.config.performance_profiling:
            args.extend(["--cov=src", "--cov-report=html", f"--cov-report=html:{self.config.output_dir}/coverage"])
        
        # Job role parametrization
        if self.config.job_roles != ["software_engineer", "ai_engineer", "data_scientist"]:
            # Filter by specific job roles (this would require custom pytest plugin or markers)
            pass
        
        # Mock LLM configuration
        if self.config.mock_llm:
            args.extend(["-k", "not requires_real_llm"])
        
        return args
    
    def _process_results(self, exit_code: int) -> None:
        """Process test results and generate summary."""
        # Parse JUnit XML results
        junit_file = Path(self.config.output_dir) / "junit_results.xml"
        
        if junit_file.exists():
            self._parse_junit_results(junit_file)
        
        # Generate summary
        total_tests = len(self.results.test_results)
        passed_tests = sum(1 for r in self.results.test_results if r.status == "passed")
        failed_tests = sum(1 for r in self.results.test_results if r.status == "failed")
        skipped_tests = sum(1 for r in self.results.test_results if r.status == "skipped")
        error_tests = sum(1 for r in self.results.test_results if r.status == "error")
        
        self.results.summary = {
            "exit_code": exit_code,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "error_tests": error_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "average_duration": sum(r.duration for r in self.results.test_results) / total_tests if total_tests > 0 else 0.0,
            "total_duration": self.results.total_duration
        }
        
        # Performance summary
        if self.config.performance_profiling:
            self._generate_performance_summary()
    
    def _parse_junit_results(self, junit_file: Path) -> None:
        """Parse JUnit XML results file."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(junit_file)
            root = tree.getroot()
            
            for testcase in root.findall(".//testcase"):
                test_name = testcase.get("name", "unknown")
                duration = float(testcase.get("time", "0"))
                
                # Determine status
                if testcase.find("failure") is not None:
                    status = "failed"
                    error_elem = testcase.find("failure")
                    error_message = error_elem.text if error_elem is not None else None
                elif testcase.find("error") is not None:
                    status = "error"
                    error_elem = testcase.find("error")
                    error_message = error_elem.text if error_elem is not None else None
                elif testcase.find("skipped") is not None:
                    status = "skipped"
                    error_message = None
                else:
                    status = "passed"
                    error_message = None
                
                result = TestResult(
                    test_name=test_name,
                    status=status,
                    duration=duration,
                    error_message=error_message
                )
                
                self.results.test_results.append(result)
        
        except Exception as e:
            print(f"⚠️  Warning: Could not parse JUnit results: {e}")
    
    def _generate_performance_summary(self) -> None:
        """Generate performance analysis summary."""
        durations = [r.duration for r in self.results.test_results if r.duration > 0]
        
        if durations:
            self.results.performance_summary = {
                "min_duration": min(durations),
                "max_duration": max(durations),
                "avg_duration": sum(durations) / len(durations),
                "total_test_time": sum(durations),
                "slowest_tests": sorted(
                    [(r.test_name, r.duration) for r in self.results.test_results],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
    
    def _generate_reports(self) -> None:
        """Generate comprehensive test reports."""
        print("\n📊 Generating Test Reports...")
        
        # JSON report
        json_file = Path(self.config.output_dir) / "e2e_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.results.to_dict(), f, indent=2, default=str)
        
        # Console summary
        self._print_summary()
        
        # Markdown report
        self._generate_markdown_report()
        
        print(f"\n📁 Reports generated in: {self.config.output_dir}")
    
    def _print_summary(self) -> None:
        """Print test summary to console."""
        summary = self.results.summary
        
        print("\n" + "="*60)
        print("🎯 E2E TEST SUMMARY")
        print("="*60)
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"⏭️  Skipped: {summary['skipped_tests']}")
        print(f"💥 Errors: {summary['error_tests']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1%}")
        print(f"⏱️  Total Duration: {summary['total_duration']:.2f}s")
        print(f"⚡ Average Test Duration: {summary['average_duration']:.2f}s")
        
        if summary['failed_tests'] > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.results.test_results:
                if result.status == "failed":
                    print(f"  • {result.test_name} ({result.duration:.2f}s)")
                    if result.error_message:
                        print(f"    Error: {result.error_message[:100]}...")
        
        if self.results.performance_summary:
            perf = self.results.performance_summary
            print("\n⚡ PERFORMANCE SUMMARY:")
            print(f"  Fastest Test: {perf['min_duration']:.2f}s")
            print(f"  Slowest Test: {perf['max_duration']:.2f}s")
            print(f"  Average Duration: {perf['avg_duration']:.2f}s")
        
        print("="*60)
    
    def _generate_markdown_report(self) -> None:
        """Generate detailed markdown report."""
        md_file = Path(self.config.output_dir) / "e2e_report.md"
        
        with open(md_file, 'w') as f:
            f.write(f"# E2E Test Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write("## Test Configuration\n\n")
            f.write(f"- **Test Suite:** {self.config.test_suite}\n")
            f.write(f"- **Job Roles:** {', '.join(self.config.job_roles)}\n")
            f.write(f"- **Parallel Workers:** {self.config.parallel_workers}\n")
            f.write(f"- **Timeout:** {self.config.timeout_seconds}s\n")
            f.write(f"- **Mock LLM:** {self.config.mock_llm}\n")
            f.write(f"- **Performance Profiling:** {self.config.performance_profiling}\n\n")
            
            # Summary
            summary = self.results.summary
            f.write("## Summary\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Tests | {summary['total_tests']} |\n")
            f.write(f"| Passed | {summary['passed_tests']} |\n")
            f.write(f"| Failed | {summary['failed_tests']} |\n")
            f.write(f"| Skipped | {summary['skipped_tests']} |\n")
            f.write(f"| Errors | {summary['error_tests']} |\n")
            f.write(f"| Success Rate | {summary['success_rate']:.1%} |\n")
            f.write(f"| Total Duration | {summary['total_duration']:.2f}s |\n")
            f.write(f"| Average Duration | {summary['average_duration']:.2f}s |\n\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            f.write("| Test Name | Status | Duration | Error |\n")
            f.write("|-----------|--------|----------|-------|\n")
            
            for result in self.results.test_results:
                status_emoji = {
                    "passed": "✅",
                    "failed": "❌",
                    "skipped": "⏭️",
                    "error": "💥"
                }.get(result.status, "❓")
                
                error_msg = result.error_message[:50] + "..." if result.error_message else "-"
                f.write(f"| {result.test_name} | {status_emoji} {result.status} | {result.duration:.2f}s | {error_msg} |\n")
            
            # Performance Analysis
            if self.results.performance_summary:
                perf = self.results.performance_summary
                f.write("\n## Performance Analysis\n\n")
                f.write(f"- **Fastest Test:** {perf['min_duration']:.2f}s\n")
                f.write(f"- **Slowest Test:** {perf['max_duration']:.2f}s\n")
                f.write(f"- **Average Duration:** {perf['avg_duration']:.2f}s\n")
                f.write(f"- **Total Test Time:** {perf['total_test_time']:.2f}s\n\n")
                
                f.write("### Slowest Tests\n\n")
                for test_name, duration in perf['slowest_tests']:
                    f.write(f"- {test_name}: {duration:.2f}s\n")


def main():
    """Main entry point for E2E test runner."""
    parser = argparse.ArgumentParser(description="Run E2E tests for CV Tailoring Application")
    
    parser.add_argument(
        "--suite",
        choices=["all", "complete_cv", "individual_item", "error_recovery"],
        default="all",
        help="Test suite to run"
    )
    
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=["software_engineer", "ai_engineer", "data_scientist"],
        help="Job roles to test (default: all)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Test timeout in seconds"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use real LLM instead of mocks (requires API keys)"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = TestRunConfig(
        test_suite=args.suite,
        job_roles=args.roles,
        parallel_workers=args.workers,
        timeout_seconds=args.timeout,
        verbose=args.verbose,
        generate_report=not args.no_report,
        output_dir=args.output_dir,
        mock_llm=not args.real_llm,
        performance_profiling=args.profile
    )
    
    # Run tests
    runner = E2ETestRunner(config)
    results = runner.run_tests()
    
    # Exit with appropriate code
    exit_code = results.summary.get("exit_code", 1)
    if results.summary.get("failed_tests", 0) > 0 or results.summary.get("error_tests", 0) > 0:
        exit_code = 1
    else:
        exit_code = 0
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()