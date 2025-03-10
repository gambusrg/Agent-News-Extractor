# News Aggregator Project

## Overview

The **News Aggregator Project** is a Python-based application that fetches and processes news articles from various sources using the GNews API. The application is designed to provide users with the latest news articles based on their queries, preprocess the text for better readability, and route messages through different agents for efficient processing.

## Features

- **Fetch News Articles**: Retrieve the latest news articles based on user-defined queries.
- **Text Preprocessing**: Clean and normalize the text by removing HTML tags and converting it to lowercase.
- **Agent-Based Architecture**: Utilize a supervisor agent to manage the flow of information between the API and preprocessing agents.
- **Logging**: Comprehensive logging to track the application's behavior and errors.

## Technologies Used

- **Python**: The primary programming language used for the application.
- **GNews API**: Used to fetch news articles from various sources.
- **LangChain**: A framework for building applications with language models.
- **Pydantic**: For data validation and settings management using Python type annotations.
- **Logging**: For tracking application events and errors.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   ```
2. Navigate to the project directory:
   ```bash
   cd YOUR_REPOSITORY
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:

```bash
python main.py
```

You can enter your query when prompted, and the application will fetch the latest news articles related to your query.

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