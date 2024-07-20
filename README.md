
# MoodLens

## Project Overview
MoodLens is an emotion detection project that classifies images into different emotion categories like happy and sad using a convolutional neural network (CNN). The project includes training data, a pre-trained model, and a web application for testing the model.

## Installation
To get started with the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/MoodLens.git
    ```

2. Navigate to the project directory:
    ```bash
    cd MoodLens
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the application, use the following command:
```bash
python app.py
```

## Project Structure
The project directory is structured as follows:
- **data/**: Contains the training images categorized into `happy` and `sad`.
- **logs/**: Contains logs from training and validation.
    - **training/**: Training logs.
    - **validation/**: Validation logs.
- **models/**: Contains the pre-trained model `imageclassifier.h5`.
- **test/**: Contains images for testing the model.
- **app.py**: The main application script.
- **requirements.txt**: Lists the dependencies required for the project.

## Contributing
Contributions are welcome! Please fork the repository and open a pull request with your changes.

## Screenshot of file structure
![image](https://github.com/user-attachments/assets/e6349f81-b156-4e8c-822b-c3db455fa87e)

