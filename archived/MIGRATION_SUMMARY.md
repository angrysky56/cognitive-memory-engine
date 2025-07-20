# Migration from requirements.txt to pyproject.toml + uv

## Summary of Changes Made

### Package Version Updates (Major Updates)
- **anthropic**: 0.3.0 → 0.57.1 (huge API improvements!)
- **chromadb**: 0.4.18 → 1.0.15 (major version with breaking changes)
- **sentence-transformers**: 2.2.2 → 5.0.0 (major version with new features)
- **openai**: 1.3.0 → 1.97.0 (massive API updates)
- **google-genai**: 0.8.0 → 1.2.0 (constrained by httpx compatibility)
- **numpy**: 1.24.0 → 2.3.1 (major version bump)
- **torch**: 2.1.0 → 2.7.1 (significant performance improvements)
- **transformers**: 4.35.0 → 4.53.2 (new models and features)

### Other Notable Updates
- **fastapi**: 0.104.0 → 0.116.1
- **uvicorn**: 0.24.0 → 0.35.0
- **pydantic**: 2.5.0 → 2.11.7
- **pytest**: 7.4.0 → 8.4.1
- **black**: 23.0.0 → 25.1.0
- **mypy**: 1.7.0 → 1.17.0
- **httpx**: constrained to <0.28 due to mcp-server-fetch

### Dependency Resolution Issues Fixed
1. **httpx version conflict**: mcp-server-fetch requires httpx<0.28, while google-genai>=1.25.0 requires httpx>=0.28.1
   - **Solution**: Made mcp-server-fetch a core dependency (as it's essential for data ingestion) and downgraded google-genai to 1.2.0
   - **Impact**: Core data ingestion functionality maintained without AI token overhead - content is fetched directly and only processed by AI when specifically requested by the user

2. **Python version requirement**: networkx 3.5 requires Python>=3.11
   - **Solution**: Updated requires-python from ">=3.10" to ">=3.11"
   - **Impact**: Modern Python features now available, better performance

### Project Structure Changes
- ✅ **pyproject.toml**: Now the single source of truth for dependencies
- ✅ **uv.lock**: Generated lockfile for reproducible builds
- 📦 **requirements.txt**: Backed up as requirements.txt.backup
- 🎯 **Data Ingestion**: mcp-server-fetch is now a core dependency for direct content fetching without AI processing overhead

## Next Steps

### 1. Update your CI/CD and documentation
```bash
# Old way (remove these from scripts)
pip install -r requirements.txt

# New way (update to this)
uv sync
# or for production
uv install --no-dev
```

### 2. Core data ingestion is now available
The system now includes mcp-server-fetch as a core dependency, enabling:
```bash
# Direct content fetching without AI processing overhead
# Content is stored raw and only processed by AI when specifically requested
```

### 3. You can safely remove requirements.txt
The requirements.txt file is no longer needed since pyproject.toml is now handling all dependencies.

### 4. Test major version updates
Some packages had major version bumps that might have breaking changes:
- **chromadb 1.0.15**: Check for API changes
- **sentence-transformers 5.0.0**: Verify model compatibility
- **numpy 2.x**: Some deprecations may affect your code

## Commands for Daily Use

```bash
# Install/sync dependencies
uv sync

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Remove a dependency
uv remove package-name

# Update all dependencies
uv update

# Run with the virtual environment
uv run python your_script.py
uv run pytest
```
