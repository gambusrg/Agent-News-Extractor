# Agent News Extractor

## Overview

The **Agent News Extractor** is a Python application designed to fetch, preprocess, and manage news articles from various sources using the GNews API. This project utilizes an agent-based architecture to efficiently handle the flow of information, allowing users to extract relevant news articles based on their queries.

## Features

- **Fetch News Articles**: Retrieve the latest news articles based on user-defined queries using the GNews API.
- **Text Preprocessing**: Clean and normalize the text by removing HTML tags and converting it to lowercase for better readability.
- **Agent-Based Architecture**: Implement a supervisor agent to manage the interaction between the API and preprocessing agents, ensuring a smooth workflow.
- **Logging**: Comprehensive logging to track application behavior and errors for easier debugging and monitoring.

## Technologies Used

- **Python**: The primary programming language for the application.
- **GNews API**: Used to fetch news articles from various sources.
- **LangChain**: A framework for building applications with language models.
- **Pydantic**: For data validation and settings management using Python type annotations.
- **Logging**: For tracking application events and errors.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone git@github.com:gambusrg/Agent-News-Extractor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Agent-News-Extractor
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
4. Install the required dependencies:
   ```bash
   pip install requirements.txt
   ```

## Usage

To run the application, execute the following command:

```bash
python main.py
```

You will be prompted to enter your query, and the application will fetch the latest news articles related to your input.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [GNews API](https://gnews.io/docs/) for providing the news data.
- [LangChain](https://langchain.readthedocs.io/en/latest/) for the framework used in building the application.
