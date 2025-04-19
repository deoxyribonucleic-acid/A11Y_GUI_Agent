# ♿️ Accessibility is All You Need
### *Utilizing Accessibility Labels as Natural Screen Captioners in GUI Agents*

## Introduction
This project leverages the Accessibility API to produce a precise segmentation and captioning map that facilitates UI understanding with significantly lower computational overhead compared to traditional OCR, object detection, or image segmentation models.

## Limitations
- **Platform Restriction:** Currently supports only 🍎 macOS. 
- **Scope Limitation:** UI extraction is confined to the foreground application with window focus. Efforts are ongoing to address this limitation on macOS.
- **Agent Complexity:** The LLM-Agent's structure is relatively rudimentary, requiring further prompt engineering and refinement of the execution flow.

## ⚡️ Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/deoxyribonucleic-acid/A11Y_GUI_Agent
   ```

2. **Install Required Dependencies:**

   ```bash
   conda create -n a11y_agent python=3.9 -y
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**

    Duplicate the `.env_template` file:

    ```bash
    cp .env_template .env
    ```
    Set up your `OPENAI_API_KEY` and `OPENAI_ORGANIZATION`. Optionally, adjust the `MODEL_NAME` if necessary:
    ```bash
    OPENAI_API_KEY="YOUR_API_KEY"
    OPENAI_ORGANIZATION="YOUR_ORGANIZATION_ID"
    MODEL_NAME="gpt-4o"
    ```

4. **Run the Agent:**

   ```bash
   python test_agent.py
   ```

## References
- [ScreenAgent](https://github.com/niuzaisheng/ScreenAgent) for system prompt design and overall system architecture.
- [OS-Copilot](https://github.com/OS-Copilot/OS-Copilot) for overall system architecture.
- [macapptree](https://github.com/MacPaw/macapptree) for accessibility label extraction.