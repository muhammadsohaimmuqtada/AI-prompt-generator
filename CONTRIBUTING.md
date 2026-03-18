# Contributing Guide

Thanks for your interest in improving AI Prompt Generator.

## Development Setup
1. Clone the repository.
2. Ensure Python 3.10+ is installed.
3. Run the app locally:
   ```bash
   python3 main.py
   ```

## Running Tests
Before opening a pull request, run:
```bash
python3 -m unittest -v
python3 -m py_compile main.py app/engine.py app/gui.py
```

## Coding Standards
- Keep code clear and maintainable.
- Prefer small, focused changes.
- Preserve existing behavior unless the change explicitly improves it.
- Add or update tests for logic changes.

## Pull Request Guidelines
- Use a descriptive PR title and summary.
- Explain what changed and why.
- Include validation steps and test output.
- Keep unrelated changes out of the same PR.

## Areas Where Contributions Help Most
- Prompt optimization quality improvements
- GUI usability and accessibility
- Model-specific template reliability
- Test coverage for edge cases

## Security and Responsible Use
This project should be used for legitimate and authorized workflows only. Contributions that improve misuse resistance and output reliability are welcome.
