# wtpsplit-triton

Universal Robust, Efficient and Adaptable Sentence Segmentation with NVIDIA Triton Support

## Installation

```bash
pip install wtpsplit-triton
```

### Install with ONNX GPU support
```bash
pip install wtpsplit-triton[onnx-gpu]
```

### Install with Triton client support
```bash
pip install wtpsplit-triton[triton]
```

## Publishing a New Release

This package uses semantic versioning and automated release workflows.

### Commit Message Format

Use conventional commits format for automatic version bumping:

- `feat:` - New feature (bumps minor version)
- `fix:` - Bug fix (bumps patch version)
- `perf:` - Performance improvement (bumps patch version)
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test changes
- `build:` - Build system changes
- `ci:` - CI configuration changes
- `chore:` - Other changes

**Breaking changes:** Add `!` after type or add `BREAKING CHANGE:` in commit body to bump major version.

Examples:
```bash
git commit -m "feat: add NVIDIA Triton inference support"
git commit -m "fix: correct attention mask dtype in Triton client"
git commit -m "feat!: change API for SaT initialization"
```

### Automated Release Process

1. **Automatic (Recommended):**
   - Push to main branch with conventional commits
   - GitHub Actions will automatically:
     - Determine version bump based on commits
     - Update version in setup.py
     - Create GitHub release
     - Publish to PyPI

2. **Manual Release:**
   - Create a git tag: `git tag v2.2.0`
   - Push tag: `git push origin v2.2.0`
   - Create GitHub release from the tag
   - GitHub Actions will publish to PyPI automatically

### PyPI Setup Required

Add `PYPI_API_TOKEN` to your repository secrets:
1. Go to https://pypi.org/manage/account/token/
2. Create new API token
3. Go to GitHub repository → Settings → Secrets and variables → Actions
4. Add new secret: `PYPI_API_TOKEN` with your token value

For trusted publishing (more secure):
1. Go to https://pypi.org/manage/account/publishing/
2. Add GitHub publisher for your repository
3. Remove `PYPI_API_TOKEN` secret (no longer needed)
