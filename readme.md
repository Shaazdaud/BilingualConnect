# README: BilingualConnect

## Overview

This project is a Flask-based web application that performs translation between English and Hindi using pre-trained models from the `ai4bharat/indictrans2` family. The application automatically detects the input language and translates it to the target language. The translation is powered by Hugging Face's `transformers` library and the `IndicTransTokenizer` for handling Indic scripts.

## Requirements

### Dependencies

The application requires the following Python libraries:

- `torch`: For running the translation models.
- `transformers`: For loading the pre-trained translation models.
- `IndicTransTokenizer`: For preprocessing and postprocessing input/output text in Indic languages.
- `Flask`: For serving the web application.

You can install the dependencies using `pip`:

```bash
pip install torch transformers indictrans-tokenizer Flask
```

### Hardware

- The application will use GPU (CUDA) if available; otherwise, it defaults to CPU.

## File Structure

```plaintext
.
├── app.py                # Main application script
├── templates
│   └── index.html        # HTML template for the web interface
└── README.md             # Project documentation
```

## How It Works

1. **Language Detection**: The application detects the input language based on the presence of Hindi characters.
2. **Translation**: Based on the detected language, the corresponding model and tokenizer are selected for translation.
3. **Rendering**: The translated text is rendered on the web page.

### Key Functions

- `translate_text(text, src_lang, tgt_lang)`: Handles the translation by loading the appropriate model and tokenizer, and returns the translated text.

- `index()`: The main route that handles both GET and POST requests. It detects the input language, translates the text, and returns the result.

## How to Run

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Run the Application**:

   ```bash
   flask run
   ```

3. **Access the Application**:
   Open a web browser and go to `http://127.0.0.1:5000/`.

## Usage

- **Input**: Enter text in either English or Hindi.
- **Output**: The application will automatically detect the language and provide the translation.

## Future Improvements

- Add support for more Indic languages.
- Implement caching to improve translation speed for frequently used phrases.
- Improve the language detection mechanism for better accuracy.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Shaazdaud