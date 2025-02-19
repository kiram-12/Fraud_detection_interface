# Fraud Detection Project

## Project Overview

The Fraud Detection Project aims to identify fraudulent transactions using advanced machine learning techniques. Our application provides an intuitive and user-friendly interface to interact with the dataset, visualize data patterns, and train various machine learning models to detect fraud effectively.

## Features

- **Data Viewing**: Easily view and navigate through the dataset.
- **Data Visualization**: Generate histograms, boxplots, scatter plots, and heatmaps for data analysis.
- **Feature Description**: Access detailed information about each feature in the dataset.
- **Model Training**: Train multiple machine learning models with various data balancing techniques.
- **Results Display**: View metrics (Accuracy,F1-score,precision,ROC curves and confusion matrices )to evaluate model performance.

## Technologies Used

- **Python**: Programming language used for development.
- **PyQt6**: For building the graphical user interface.
- **Pandas**: For data manipulation and analysis.
- **Seaborn and Matplotlib**: For data visualization.
- **Scikit-learn**: For implementing machine learning models and techniques.
- **Custom Python Scripts**: For data preprocessing and model training functions.

## Installation

To install and run the application, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/kiram-12/Fraud_detection_interface.git
    cd fraud-detection-project
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**:
    ```bash
    python codepyqt6.py
    ```

## Usage

1. **Launch the Application**:
    - Run `python codepyqt6.py` to start the application.
    - The welcome page will appear with an introductory message and an "Enter" button.

2. **Navigate the Interface**:
    - Use the sidebar on the left to navigate between different sections: Home, View Data, Visualize Data, Description of Features, and Launch Training.

3. **View Data**:
    - Click on "View Data" to see the dataset in a table format.

4. **Visualize Data**:
    - Select features and generate histograms, boxplots, scatter plots, and heatmaps to analyze data patterns.

5. **Feature Description**:
    - Access detailed information about each feature in the dataset, including statistical summaries.

6. **Train Models**:
    - Choose from multiple machine learning models and data balancing techniques.
    - Click on "Start Training" to train the model and view results.

7. **Evaluate Results**:
    - View ROC curves and confusion matrices to evaluate model performance.

## Project Structure

- **main.py**: Entry point of the application.
- **ui/**: Contains the user interface files.
- **data/**: Contains the dataset.
- **visualization/**: Scripts for data visualization.
- **models/**: Scripts for training machine learning models.
- **resources/**: Contains QSS files for theming and other resources.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact us at [ikramnmili12@gmail.com].

---

Thank you for using our Fraud Detection application! We hope it helps you effectively detect and prevent fraudulent transactions.
