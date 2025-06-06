# Post-Reorganization Checklist

## ✅ Completed
- [x] Git backup created
- [x] Directory structure created
- [x] Files moved to new locations
- [x] __init__.py files created
- [x] Placeholder files created

## 🔧 Required Updates

### 1. Import Statement Updates
The following files need their import statements updated:

#### Core Files
- `src/core/main.py` - Update imports for agents, services, utils
- `src/core/orchestrator.py` - Update imports for agents and services
- `src/core/state_manager.py` - Update any relative imports

#### Agent Files (in `src/agents/`)
- All agent files need to update imports for:
  - `agent_base.py` → `from src.agents.agent_base import AgentBase`
  - LLM service → `from src.services.llm import LLM`
  - Other utilities and services

#### Service Files
- `src/services/llm.py` - Update any imports
- `src/services/vector_db.py` - Update any imports

#### Utility Files
- `src/utils/template_manager.py` - Update template paths
- `src/utils/template_renderer.py` - Update template paths

#### API Files
- `src/api/main.py` - Update all imports to new structure
- `src/api/app_main.py` - Update all imports to new structure

#### Test Files
- All test files in `tests/unit/` need import updates

### 2. Configuration File Updates

#### requirements.txt
- No changes needed (already in root)

#### Dockerfile
- Update COPY commands to reflect new structure
- Update WORKDIR and entry points

#### .gitignore
- Add any new directories that should be ignored
- Update paths if needed

#### .vscode/settings.json
- Update Python path settings
- Update test discovery paths

### 3. Path References

#### Template Paths
- Update hardcoded paths in template manager
- Update paths in configuration files
- Update paths in documentation

#### Data Paths
- Update session data paths
- Update prompt file paths
- Update job description paths

### 4. Testing

#### Unit Tests
- Run all unit tests: `python -m pytest tests/unit/`
- Fix any failing tests due to import issues
- Update test data paths

#### Integration Tests
- Create integration tests in `tests/integration/`
- Test end-to-end workflows

### 5. Documentation Updates

#### README.md
- Update project structure section
- Update installation instructions
- Update usage examples with new paths

#### Development Documentation
- Update architecture diagrams
- Update file organization documentation
- Update contribution guidelines

### 6. Deployment Updates

#### Docker
- Test Docker build with new structure
- Update docker-compose if exists

#### CI/CD
- Update any CI/CD scripts
- Update deployment scripts

## 🚀 Execution Steps

### Step 1: Update Import Statements
```bash
# Run the import update script (to be created)
python scripts/update_imports.py
```

### Step 2: Update Configuration Files
```bash
# Update Dockerfile
# Update any deployment scripts
```

### Step 3: Run Tests
```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests
python -m pytest tests/unit/ -v

# Run specific test modules
python -m pytest tests/unit/test_agent_base.py -v
```

### Step 4: Test Application
```bash
# Test main application
python src/core/main.py

# Test API if applicable
python src/api/main.py
```

### Step 5: Update Documentation
- Review and update README.md
- Update any API documentation
- Update development guides

### Step 6: Final Commit
```bash
# Add all changes
git add .

# Commit the reorganized structure
git commit -m "Reorganize project structure - move to src/ based layout

- Move core files to src/core/
- Move agents to src/agents/
- Move services to src/services/
- Move utilities to src/utils/
- Move templates to src/templates/
- Move API files to src/api/
- Reorganize data and documentation
- Update import statements
- Update configuration files"
```

## 🔍 Verification Checklist

- [ ] All Python files can be imported without errors
- [ ] All tests pass
- [ ] Main application runs successfully
- [ ] API endpoints work (if applicable)
- [ ] Docker build succeeds
- [ ] Documentation is updated
- [ ] No broken relative imports
- [ ] Template paths are correct
- [ ] Data file paths are correct
- [ ] Configuration files are updated

## 📝 Notes

- The reorganization moved files from a flat structure to a proper Python package structure
- All agent files are now in `src/agents/`
- Core application logic is in `src/core/`
- Services (LLM, vector DB) are in `src/services/`
- Utilities are in `src/utils/`
- Templates are in `src/templates/`
- Data files are organized in `data/` with subdirectories
- Tests are separated into unit and integration
- Documentation is organized in `docs/`

## 🚨 Common Issues to Watch For

1. **Circular Imports**: Check for circular dependencies between modules
2. **Path Issues**: Ensure all file paths are updated correctly
3. **Test Discovery**: Make sure pytest can discover all tests
4. **Template Loading**: Verify template files can be found
5. **Data File Access**: Ensure data files are accessible from new locations
6. **Configuration Loading**: Check that config files are loaded correctly