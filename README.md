# multimodal RAG-PyMuPDF4LLM-llamaindex-Qdrant

## Overview

The PDF Content Extraction and Retrieval App allows users to upload PDF documents, extract their content, and perform queries to retrieve relevant information, including images. Built using Streamlit, this application leverages powerful Python libraries such as PyMuPDF4LLM, Qdrant, and LlamaIndex to provide an intuitive user experience for managing PDF content.

## Features

- Upload PDF files easily.
- Convert PDF content into structured Markdown format.
- Extract images from PDFs and store them for easy access.
- Query extracted content to retrieve relevant text and images.
- User-friendly interface with a clean design.

## Technologies Used

- **Streamlit**: For building the web application interface.
- **PyMuPDF4LLM**: For handling PDF files and extracting content.
- **Qdrant**: For storing and retrieving vector representations of text and images.
- **LlamaIndex**: For managing documents and indices efficiently.
- **Matplotlib**: For visualizing retrieved images.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/toni-ramchandani/multimodal-PyMuPDF4LLM-llamaindex-Qdrant.git
   cd pdf-extraction-app
   pip install -r requirements.txt
   streamlit run app.py

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements or new features.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
