# CHATCYBER-TKINTER

Cyber Assistant v4.0 is a cybersecurity assistance tool developed in Python that combines natural language processing, sentiment analysis, voice recognition, text-to-speech, and image metadata extraction. The tool features a graphical user interface (Tkinter) for interactive chatting and online investigations.

## Features

- **Interactive Chat:** Text-based communication with responses generated by a language model.
- **Voice Interaction:** Voice command recognition and text-to-speech for audible responses.
- **Online Investigation:** Searches and analyzes online data using DuckDuckGo, including news and leaked data.
- **Image Metadata Analysis:** Extracts and displays EXIF metadata from image URLs.
- **Sentiment and Language Analysis:** Evaluates sentiment and language detection using NLTK and langdetect.
- **Customizable Interface:** Options to change language, response style, and investigation focus.

## Dependencies

To run the tool, you need Python 3.7+ and the following libraries:

```bash
pip install llama-cpp-python huggingface_hub duckduckgo_search requests psutil nltk langdetect cachetools emoji SpeechRecognition pyttsx3 Pillow
Additional Notes:

PyAudio: May be required for voice recognition. Depending on your system, install it via pip install PyAudio or use platform-specific installers.

Language Model: The project uses the "Mistral-7B-Instruct-v0.1-GGUF" model. If the model file is not available locally, the system will attempt to download it automatically using the huggingface_hub.

Initial Setup
Clone or copy the source code to your local machine.

Install the dependencies as listed above.

Run the main script:

bash
Copiar
Editar
python your_script.py
Usage
Interface: The main window has two tabs: Chat and Investigation.

In the Chat tab, you can converse with the assistant, which will respond in either a technical or creative manner based on your configuration.

In the Investigation tab, the tool performs online searches and generates detailed reports, also displaying links in a table.

Voice Commands: Toggle voice recognition to send commands and receive spoken responses.

Metadata Extraction: Enter an image URL in the designated field and click "Analyze" to view the EXIF metadata.

Troubleshooting
Model Download: Check your internet connection if there are issues with the automatic download of the model.

Voice Recognition: Ensure your microphone is configured correctly and that PyAudio is installed.

Dependencies: If you encounter "module not found" errors, verify that all libraries have been installed correctly.

License
This project is provided "as is," without warranties or formal support. Use and modify it according to your needs.

Contact
For questions, suggestions, or issues, please contact the project maintainer.

yaml
Copiar
Editar

---

**guide.txt**

```txt
Cyber Assistant v4.0 - Quick Start Guide

1. Introduction:
   - A cybersecurity assistance tool with chat, voice commands, online investigations, and image metadata analysis.
   - Developed in Python with a graphical interface (Tkinter).

2. Requirements:
   - Python 3.7 or higher.
   - Install the following dependencies:
     pip install llama-cpp-python huggingface_hub duckduckgo_search requests psutil nltk langdetect cachetools emoji SpeechRecognition pyttsx3 Pillow
   - PyAudio might be required for voice recognition.

3. Execution:
   - Run the main script:
     python your_script.py
   - Use the "Chat" tab for conversation and the "Investigation" tab for online research.

4. Features:
   - Responses generated by a language model.
   - Voice commands for interaction.
   - Detailed online investigation reports.
   - Extraction of image metadata via URL.
   - Sentiment analysis and language detection.

5. Configuration:
   - Select language and response style (Technical or Creative).
   - Define the investigation focus and configure searches for sites, news, and leaked data.
   - Enable/disable image metadata analysis as needed.

6. Tips:
   - Ensure your microphone and audio drivers are properly set up.
   - Verify your internet connection for automatic model downloads and updates.
   - Refer to README.md for detailed information and troubleshooting.
