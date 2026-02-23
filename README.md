# Research Agents

Will update soon with an architecture overview.

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

To deactivate when you're done:
```bash
deactivate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Copy the example env file:

```bash
cp .env.example .env
```

Then edit `.env` with your provider and key. You only need the key for the provider you're using:

**OpenAI:**
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
```

**Anthropic:**
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

**Google Gemini:**
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...
```

Each provider comes with sensible default models (fast/general/strong tiers). You can override them in `.env` with `MODEL_FAST`, `MODEL_GENERAL`, and `MODEL_STRONG`.

**Important:** Never commit your `.env` file. It is already in `.gitignore`.
