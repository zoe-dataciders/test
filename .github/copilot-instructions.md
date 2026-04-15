---
description: "Python project guidelines: always check for missing dependencies, prefer modular code structure, and validate after changes"
applyTo: "**/*.py"
---

# Python Development Instructions

## Dependency Management
- Always check for missing dependencies when reviewing or running Python code.
- If dependencies are not installed, create a `requirements.txt` file and install them using `pip install -r requirements.txt`.
- For optional dependencies (e.g., machine learning libraries), handle import errors gracefully and inform the user.

## Code Structure
- Prefer modular code: split large files into logical modules (e.g., separate files for models, utilities, main logic).
- Use relative imports within the project for better organization.
- Ensure the main entry point is clear and runnable from the command line.

## Validation
- After any substantive changes (e.g., editing code, installing deps), run tests or basic validation to ensure the code works.
- For runnable scripts, test with minimal inputs to confirm functionality.
- Point out potential issues like syntax errors, missing imports, or logical bugs.

## Best Practices
- Use descriptive filenames that reflect the code's purpose.
- Include docstrings and comments for complex logic, especially for beginners.
- Follow Python conventions (PEP 8) for readability.

These instructions apply to all Python files in the workspace to maintain consistent, maintainable code.