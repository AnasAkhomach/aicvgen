# Project Reorganization Script
# This script moves existing files to the new directory structure

$ProjectRoot = "c:\Users\Nitro\Desktop\aicvgen"
Set-Location $ProjectRoot

Write-Host "Starting project reorganization..." -ForegroundColor Yellow

# Create new directory structure
$directories = @(
    "src\core",
    "src\agents",
    "src\services",
    "src\utils",
    "src\models",
    "src\config",
    "src\templates",
    "src\api",
    "src\frontend\static",
    "src\frontend\templates",
    "data\prompts",
    "data\job_descriptions",
    "data\sessions",
    "data\templates",
    "tests\unit",
    "tests\integration",
    "docs\dev",
    "docs\user",
    "scripts",
    "config"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $ProjectRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# Move core files
Write-Host "Moving core files..." -ForegroundColor Cyan
Move-Item "main.py" "src\core\main.py" -Force
Move-Item "orchestrator.py" "src\core\orchestrator.py" -Force
Move-Item "state_manager.py" "src\core\state_manager.py" -Force
Move-Item "llm.py" "src\services\llm.py" -Force
Move-Item "vector_db.py" "src\services\vector_db.py" -Force

# Move agent files
Write-Host "Moving agent files..." -ForegroundColor Cyan
Move-Item "agent_base.py" "src\agents\agent_base.py" -Force
Move-Item "content_writer_agent.py" "src\agents\content_writer_agent.py" -Force
Move-Item "cv_analyzer_agent.py" "src\agents\cv_analyzer_agent.py" -Force
Move-Item "formatter_agent.py" "src\agents\formatter_agent.py" -Force
Move-Item "parser_agent.py" "src\agents\parser_agent.py" -Force
Move-Item "quality_assurance_agent.py" "src\agents\quality_assurance_agent.py" -Force
Move-Item "research_agent.py" "src\agents\research_agent.py" -Force
Move-Item "tools_agent.py" "src\agents\tools_agent.py" -Force
Move-Item "vector_store_agent.py" "src\agents\vector_store_agent.py" -Force

# Move utility files
Write-Host "Moving utility files..." -ForegroundColor Cyan
Move-Item "template_manager.py" "src\utils\template_manager.py" -Force
Move-Item "template_renderer.py" "src\utils\template_renderer.py" -Force

# Move template files
Write-Host "Moving template files..." -ForegroundColor Cyan
Move-Item "cv_template.md" "src\templates\cv_template.md" -Force
Move-Item "tailored_cv.md" "src\templates\tailored_cv.md" -Force
Move-Item "Anas_Akhomach-main-template-en.md" "src\templates\Anas_Akhomach-main-template-en.md" -Force

# Move API files
Write-Host "Moving API files..." -ForegroundColor Cyan
if (Test-Path "app") {
    Move-Item "app\api\main.py" "src\api\main.py" -Force
    Move-Item "app\main.py" "src\api\app_main.py" -Force
    if (Test-Path "app\frontend") {
        if (Test-Path "app\frontend\static") {
            Get-ChildItem "app\frontend\static" | Move-Item -Destination "src\frontend\static" -Force
        }
        if (Test-Path "app\frontend\templates") {
            Get-ChildItem "app\frontend\templates" | Move-Item -Destination "src\frontend\templates" -Force
        }
    }
    Remove-Item "app" -Recurse -Force
}

# Move data files
Write-Host "Moving data files..." -ForegroundColor Cyan
if (Test-Path "prompts_folder") {
    Get-ChildItem "prompts_folder" -Recurse | Move-Item -Destination "data\prompts" -Force
    Remove-Item "prompts_folder" -Recurse -Force
}

if (Test-Path "job_desc_folder") {
    Get-ChildItem "job_desc_folder" -Recurse | Move-Item -Destination "data\job_descriptions" -Force
    Remove-Item "job_desc_folder" -Recurse -Force
}

# Sessions data is already in the correct location

# Move documentation
Write-Host "Moving documentation files..." -ForegroundColor Cyan
if (Test-Path "dev_docs") {
    Get-ChildItem "dev_docs" | Move-Item -Destination "docs\dev" -Force
    Remove-Item "dev_docs" -Recurse -Force
}

# Move tests
Write-Host "Moving test files..." -ForegroundColor Cyan
if (Test-Path "tests") {
    Get-ChildItem "tests" | Move-Item -Destination "tests\unit" -Force
}

# Move configuration files to config directory
Write-Host "Moving configuration files..." -ForegroundColor Cyan
if (Test-Path ".pylintrc") {
    Move-Item ".pylintrc" "config\.pylintrc" -Force
}

# Create __init__.py files
Write-Host "Creating __init__.py files..." -ForegroundColor Cyan
$initDirs = @(
    "src",
    "src\core",
    "src\agents",
    "src\services",
    "src\utils",
    "src\models",
    "src\config",
    "src\templates",
    "src\api",
    "src\frontend",
    "tests",
    "tests\unit",
    "tests\integration"
)

foreach ($dir in $initDirs) {
    $initFile = Join-Path $ProjectRoot "$dir\__init__.py"
    if (-not (Test-Path $initFile)) {
        New-Item -ItemType File -Path $initFile -Force | Out-Null
    }
}

# Create placeholder files for empty directories
Write-Host "Creating placeholder files..." -ForegroundColor Cyan
$placeholders = @(
    "src\models\.gitkeep",
    "src\config\.gitkeep",
    "tests\integration\.gitkeep",
    "docs\user\.gitkeep",
    "scripts\.gitkeep",
    "data\templates\.gitkeep"
)

foreach ($placeholder in $placeholders) {
    $placeholderPath = Join-Path $ProjectRoot $placeholder
    if (-not (Test-Path $placeholderPath)) {
        New-Item -ItemType File -Path $placeholderPath -Force | Out-Null
    }
}

Write-Host "Project reorganization completed successfully!" -ForegroundColor Green
Write-Host "\nNext steps:" -ForegroundColor Yellow
Write-Host "1. Update import statements in Python files" -ForegroundColor White
Write-Host "2. Update configuration files (requirements.txt, Dockerfile, etc.)" -ForegroundColor White
Write-Host "3. Run tests to ensure everything works" -ForegroundColor White
Write-Host "4. Update documentation" -ForegroundColor White
Write-Host "5. Commit the reorganized structure" -ForegroundColor White