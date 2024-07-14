# Nlp-Based-MailSearch

## Description
This project implements an email search application that uses BERT-based semantic search for efficient retrieval of emails based on natural language queries. It parses email datasets, processes text using BERT embeddings, and enables querying based on sender, receiver, date, and subject.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Example](#Example)
- [Directory Structure](#Directory-Structure)
- [Features](#Features)
- [Dependencies](#Dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation
1. Folk and clone the repository:
   ```
   git clone https://github.com/YourUsername/Nlp-Based-MailSearch.git
   cd YourRepository
   ```

2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure SQL Server (SQL Express) is installed and running on your system.

## Usage
1. Set up your database credentials in the `app.py` file:
   ```python
   connection_string = (
       "Driver={ODBC Driver 17 for SQL Server};"
       "Server=localhost\\SQLEXPRESS;"
       "Database=ENRON_Emails;"
       "UID='YOUR_UID';"
       "PWD='YOUR_PASSWORD';"
   )
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate
   ```
   On Windows:
   ```sh
   python -m venv venv
   source venv\Scripts\Activate
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Enter your search query in the Streamlit app interface to retrieve relevant emails.

## Example
1. Enter a natural language query in the input box (e.g., "Find emails from John to Jane last week about the project update").
2. View search results with relevant emails displayed.

## Directory Structure
```
- README.md           # Project documentation
- app.py              # Streamlit application for email search
- requirements.txt    # Required Python packages
- small_enron.csv     # Sample dataset
- EmailSearch.py      # Class definition for email searching
```

## Features
- BERT-based semantic search for email contents.
- Query emails by sender, recipient, date, and subject.
- Efficient retrieval using SQL queries.
- Streamlit interface for user interaction.

## Dependencies
- pandas
- streamlit
- gmail (custom module)
- transformers
- torch
- pyodbc
- numpy
- sklearn
- nltk
- spacy

## Contributing
Contributions are welcome. Please fork the repository, create a new branch, make your changes, and submit a pull request.

## License
This project is licensed under the Apache License Version 2.0. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or support, please contact us.