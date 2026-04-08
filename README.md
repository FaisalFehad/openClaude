# openClaude

Use any OpenAI-compatible model with [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

openClaude is a lightweight proxy that translates between the Anthropic Messages API and the OpenAI Chat Completions API. Point Claude Code at it and use models from DeepInfra, Together, OpenRouter, Groq, Fireworks, Moonshot, or any OpenAI-compatible provider.

Native Claude models pass through to Anthropic's API directly — no configuration needed for those.

Built with Go. Zero dependencies. Single binary.

## How it works

Claude Code speaks the Anthropic Messages API. openClaude sits in between, translating:

```
Claude Code  →  Anthropic Messages API  →  openClaude  →  OpenAI Chat Completions API  →  Provider
```

It handles message conversion, tool calling, streaming SSE translation, vision/images, and thinking/reasoning content from models like DeepSeek-R1 and QwQ.

## Getting started

### 1. Install

```bash
git clone https://github.com/FaisalFehad/openClaude.git
cd openClaude
./install.sh
```

Requires [Go](https://go.dev/dl/) and [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

### 2. Configure

Edit `~/.config/openClaude/config.json` (created by the installer from `config.example.json`):

```json
{
  "port": 8082,
  "providers": {
    "deepinfra": {
      "url": "https://api.deepinfra.com/v1/openai/chat/completions",
      "apiKeyEnv": "DEEPINFRA_API_KEY"
    }
  },
  "models": {
    "qwen3-max": "deepinfra/Qwen/Qwen3-Max"
  }
}
```

Each provider needs a `url` and an API key via either:
- `apiKeyEnv` — name of an environment variable
- `apiKeyCmd` — a shell command (e.g. macOS Keychain: `security find-generic-password -a deepinfra -s deepinfra-api-key -w`)

Models use the format `provider/model-id`. The provider name references a key in `providers`.

### 3. Run

```bash
openClaude --model qwen3-max
```

This starts the proxy and launches Claude Code in one command.

Other commands:

```bash
openClaude list      # show configured models
openClaude serve     # start proxy only (without launching Claude Code)
```

## Uninstall

```bash
./uninstall.sh
```
