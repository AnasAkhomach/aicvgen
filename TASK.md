#### CB-015
- **ID**: CB-015
- **Category**: CONTRACT_BREACH
- **Location**: `src/integration/` (multiple files)
- **Summary**: Integration layer contracts allow bypassing of service layer boundaries
- **Rationale**: Direct access to core services violates architectural contracts
- **Remediation Plan**: Enforce service layer contracts through proper interface definitions
- **Effort Estimate**: High