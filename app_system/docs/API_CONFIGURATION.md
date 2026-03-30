# API Configuration Guide

## Overview

This project uses a `.env` file to manage API credentials securely. API keys and endpoints are never committed to version control.

## Setup for New Users

### 1. Copy the Template
```bash
cd app_system
cp .env.example .env
```

### 2. Edit Configuration
Open `.env` in your editor and fill in your credentials:
```bash
nano .env  # or use your preferred editor
```

### 3. Required Settings
```env
API_PROVIDER=your_provider
API_KEY=your_api_key_here
API_BASE=https://api.example.com/v1
MODEL_PRIMARY=your_primary_model
MODEL_SECONDARY=your_secondary_model
MODEL_TERTIARY=your_tertiary_model
```

## Supported API Providers

### OpenAI
```env
API_PROVIDER=openai
API_KEY=sk-...
API_BASE=https://api.openai.com/v1
MODEL_PRIMARY=gpt-4
MODEL_SECONDARY=gpt-4
MODEL_TERTIARY=gpt-3.5-turbo
```

### Anthropic
```env
API_PROVIDER=anthropic
API_KEY=sk-ant-...
API_BASE=https://api.anthropic.com/v1
MODEL_PRIMARY=claude-sonnet-4
MODEL_SECONDARY=claude-opus-4
MODEL_TERTIARY=claude-sonnet-3-5
```

### Google Gemini
```env
API_PROVIDER=gemini
API_KEY=AIza...
API_BASE=https://generativelanguage.googleapis.com/v1
MODEL_PRIMARY=gemini-pro
MODEL_SECONDARY=gemini-pro
MODEL_TERTIARY=gemini-pro
```

### Custom/Internal API
```env
API_PROVIDER=custom
API_KEY=your-key
API_BASE=https://your-internal-api.example.com/v1
MODEL_PRIMARY=your-model-id
MODEL_SECONDARY=your-model-id-2
MODEL_TERTIARY=your-model-id-3
```

## How It Works

1. **config.py** loads settings from `.env` file (or environment variables)
2. **utils.py** imports credentials from config.py
3. **.env** is gitignored - never committed to repo
4. **.env.example** provides a template for others

## File Structure

```
app_system/
├── config.py          # Configuration loader
├── .env               # Your secrets (gitignored)
├── .env.example       # Template for others
└── utils.py           # Imports from config
```

## Security Best Practices

### ✅ DO
- Keep `.env` file private
- Use different API keys for different environments
- Copy `.env.example` to `.env` for local setup
- Use environment variables in production

### ❌ DON'T
- Commit `.env` to version control
- Share your API keys in code
- Hardcode credentials anywhere
- Push `.env` to GitHub/GitLab

## Testing Your Configuration

```bash
cd app_system

# Test config loading
python config.py

# Should output:
# ✓ Loaded configuration from /path/to/.env
# Configuration summary with masked API key
```

## Troubleshooting

### Error: "Required environment variable 'API_KEY' is not set!"
- Make sure you created `.env` file
- Check that `.env` has `API_KEY=...` line
- Verify you're in the `app_system/` directory

### Error: "python-dotenv not installed"
```bash
pip install python-dotenv
```

### Verify .env is gitignored
```bash
cd /path/to/research_agents
git check-ignore app_system/.env
# Should output: app_system/.env
```

## Model Selection Guide

### MODEL_PRIMARY
Used by Section Evaluator for detailed analysis. Requires:
- Strong reasoning capabilities
- JSON output support
- Good at structured evaluation

**Recommended**: Claude Sonnet 4, GPT-4

### MODEL_SECONDARY
Used by Referee Debate System with thinking mode. Requires:
- Extended context window
- Debate/argumentation skills
- Consistency across rounds

**Recommended**: Claude Opus 4, Claude Sonnet 3.7

### MODEL_TERTIARY
Legacy/backup model for fallback scenarios.

**Recommended**: Claude Sonnet 3.5, GPT-3.5-turbo

---

**Last Updated**: 2026-03-30
**Status**: Production
