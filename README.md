# DeepSeek CLI + PyQt5 UI (DS_KEY)

Minimal Python examples that connect to DeepSeek official API and answer simple questions.

- CLI: `deepseek_cli.py` (single-turn chat)
- Function layer: `deepseek_api.py` (core API logic)
- System prompt manager: `prompts_manager.py` (typed persistence layer)
- PyQt5 UI wrapper: `deepseek_pyqt5_ui.py` (multi-turn chat with system prompts)

## 1) Set your DS_KEY environment variable

PowerShell (current terminal session):

```powershell
$env:DS_KEY = "your_deepseek_api_key"
```

PowerShell (persist for current user):

```powershell
[System.Environment]::SetEnvironmentVariable("DS_KEY", "your_deepseek_api_key", "User")
```

After setting persistent env vars, restart terminal/VS Code.

## 2) Install PyQt5 (for GUI)

```powershell
pip install -r requirements.txt
```

## 3) Run CLI

```powershell
python deepseek_cli.py
```

Single-turn simple Q&A. Type questions one at a time. Type `exit` or `quit` to stop.

## 4) Run PyQt5 UI

```powershell
python deepseek_pyqt5_ui.py
```

### Features

- **Model switch**: Flash (`deepseek-v4-flash`) / Pro (`deepseek-v4-pro`)
- **Multi-turn chat**: Full conversation history preserved across messages
- **System prompt management**:
  - Pre-built templates: "default", "code_expert", "translator", "writer"
  - Create/delete custom prompts
  - Edit prompt content directly in the text box (no separate Edit button)
  - Prompts are stored by type in local files in current folder (for example `prompts_system.json`, `prompts_roleplay.json`, `prompts_tool.json`)
  - Dialog close automatically saves latest local prompt updates
- **Streaming output**: Real-time token display for faster feedback
- **Stop button**: Interrupt long-running streams on demand
- **Clear history**: Reset conversation and start fresh

### Usage

1. Click **"System Prompts..."** to open the prompt manager
  - Select a saved prompt, or click **"New"** to create an empty prompt entry
  - Edit content directly in **"Prompt Content"** text box
   - Click **"Use (Close Dialog)"** to activate
2. Type a message in **"Your Message"** and click **"Send"**
3. For multi-turn conversation, just keep typing follow-up messages
4. Check **"Stream output"** to see tokens as they arrive (uncheck for batched response)
5. Click **"Stop"** anytime to cancel an in-progress request
6. Click **"Clear History"** to reset the conversation

## Notes

- Uses `DS_KEY` from environment variables.
- Calls DeepSeek official endpoint: `https://api.deepseek.com/chat/completions`.
- Model mapping from official docs:
  - Flash mode → `deepseek-v4-flash`
  - Pro mode → `deepseek-v4-pro`
- System prompts are passed once per API call (role="system")
- Conversation history automatically included in multi-turn chats
- Interruption is non-blocking: UI remains responsive
