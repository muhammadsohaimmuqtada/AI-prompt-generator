# AI Prompt Generator

A local, model-aware prompt engineering studio that rewrites rough user input into clear, high-performance prompts for major LLM families.

## Why This Project
Most users write incomplete or messy prompts. This tool improves prompt quality automatically by adding structure, intent clarity, and output constraints while preserving what the user actually wants.

## Key Features
- Model-specific prompt builders for:
  - OpenAI-style chat models
  - Anthropic Claude
  - Google Gemini
  - DeepSeek
- Smart optimization pipeline:
  - Role normalization
  - Action expansion for weak prompts
  - Context hardening and scope framing
  - Expectation cleanup and conflict resolution
  - Few-shot example scaffolding
- Prompt quality controls:
  - Auto-enhance generated instructions
  - Smart prompt optimization
  - Clarifying-question mode
  - Planning mode
  - Self-check mode
- Built-in prompt readiness report:
  - Overall score
  - Field-level scores (Role/Action/Context/Expectation/Examples)
  - Suggestions and auto-upgrades applied
- Safety hardening for bypass-style wording (normalized to policy-compliant language)

## Privacy
- Runs locally with a desktop GUI (`tkinter`)
- No built-in telemetry or analytics
- No automatic network calls for prompt generation
- Clipboard copy is user-triggered only

## Project Structure
```text
AI-prompt-generator/
├── app/
│   ├── engine.py      # Prompt optimization + model-specific builders
│   ├── gui.py         # Tkinter desktop UI
│   └── __init__.py
├── tests/
│   └── test_engine.py
├── main.py
└── run.sh
```

## Requirements
- Python 3.10+
- Standard library only (no extra pip dependencies required)

## Quick Start
```bash
# from project root
python3 main.py
```

Or:
```bash
chmod +x run.sh
./run.sh
```

## Running Tests
```bash
python3 -m unittest -v
```

## Usage Flow
1. Select your target engine.
2. Fill Role, Action, Context, Expectation, and optional Examples.
3. Keep smart optimization options enabled for best results.
4. Click **Generate Prompt**.
5. Copy the compiled payload into your target model.

## Intended Use
This project is designed to improve prompt clarity and output quality for legitimate workflows (development, analysis, writing, research, and authorized security testing).

## License
This project is licensed under the MIT License. See [LICENSE](./LICENSE).
