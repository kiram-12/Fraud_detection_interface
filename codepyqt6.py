import sys
import os
from io import StringIO
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QComboBox, QStackedWidget, QTextEdit, QTableWidget, QTableWidgetItem, QFileDialog, QListWidget, QAbstractItemView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
import pandas as pd
from Classification_Technics import DT_modele, SVM_modele, KNN_modele, Test_model
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import numpy as np

if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # sets the sys.frozen attribute and this is the directory where
    # the bundled files are stored.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the files
icon_path = os.path.join(application_path, 'icon.jpeg')
image_path = os.path.join(application_path, 'image.webp')
qss_path = os.path.join(application_path, 'design.qss')
csv_path = os.path.join(application_path, 'creditcard.csv')
class ModelTrainingThread(QThread):
    training_completed = pyqtSignal(pd.DataFrame)
    roc_data = pyqtSignal(object, object, float)
    confusion_matrix_data = pyqtSignal(object)

    def __init__(self, X, y, model_func, balance_technique):
        super().__init__()
        self.X = X
        self.y = y
        self.model_func = model_func
        self.balance_technique = balance_technique

    def run(self):
        try:
            scores, fpr, tpr, auc, cm = Test_model(self.X, self.y, self.model_func, self.balance_technique)
            self.training_completed.emit(scores)
            self.roc_data.emit(fpr, tpr, auc)
            self.confusion_matrix_data.emit(cm)
        except ValueError as e:
            print(f"Error during model training: {e}")
            self.training_completed.emit(pd.DataFrame())
            self.roc_data.emit([], [], 0.0)
            self.confusion_matrix_data.emit(np.array([]))

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1000, 800)
        self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("Machine Learning Interface")

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Sidebar
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(0, 0, 0, 0)
        sidebar.setSpacing(0)

        sidebar_frame = QFrame()
        sidebar_frame.setLayout(sidebar)
        sidebar_frame.setFrameShape(QFrame.Shape.StyledPanel)
        sidebar_frame.setProperty("class", "sidebar")
        sidebar_frame.setMinimumWidth(120)
        sidebar_frame.setMaximumWidth(160)

        home_button = QPushButton("Home")
        home_button.setProperty("class", "sidebar-btn")
        home_button.clicked.connect(self.show_home)
        sidebar.addWidget(home_button)
        
        self.view_data_button = QPushButton("Data")
        self.view_data_button.setProperty("class", "sidebar-btn")
        self.view_data_button.clicked.connect(self.view_data)
        sidebar.addWidget(self.view_data_button)

        self.visualize_button = QPushButton("Visualize Data")
        self.visualize_button.setProperty("class", "sidebar-btn")
        self.visualize_button.clicked.connect(self.show_visualization_options)
        sidebar.addWidget(self.visualize_button)
        
        self.description_button = QPushButton("Description of Features")
        self.description_button.setProperty("class", "sidebar-btn")
        self.description_button.clicked.connect(self.show_description)
        sidebar.addWidget(self.description_button)
        
        self.training_button = QPushButton("Launch Training")
        self.training_button.setProperty("class", "sidebar-btn")
        self.training_button.clicked.connect(self.show_training_options)
        sidebar.addWidget(self.training_button)

      
        
        sidebar.addStretch()

        main_layout.addWidget(sidebar_frame)

        # Content area
        self.content_layout = QStackedWidget()
        main_layout.addWidget(self.content_layout)

        # Home content
        home_content = QWidget()
        home_layout = QVBoxLayout(home_content)
        home_layout.setContentsMargins(20, 20, 20, 20)
        home_layout.setSpacing(15)
      

        intro_text = ("<h2>Welcome to our Fraud Detection Project</h2>"
"<p>This project uses machine learning techniques to detect fraudulent transactions in credit card data.</p>"
"<p>We use various balancing techniques to handle imbalanced data before training our machine learning models.</p>"
"<h3>Features:</h3>"
"<ul>"
"<li><b>Visualize Data:</b> Display graphs and descriptive statistics of the data.</li>"
"<li><b>View Data:</b> Display all data in a table.</li>"
"<li><b>Launch Training:</b> Select and train a machine learning model with different balancing techniques.</li>"
"<li><b>Show Results:</b> View cross-validation scores for trained models and the ROC curve with AUC.</li>"
"</ul>"
"<h3>Techniques Used:</h3>"
"<p>We have used the following techniques for data balancing:</p>"
"<ul>"
"<li><b>Random Undersampling:</b> Random sampling to reduce the number of majority class examples.</li>"
"<li><b>NearMiss:</b> Selection of majority class examples based on their proximity to minority class examples.</li>"
"<li><b>Tomek Links:</b> Removal of pairs of different class examples that are mutually close.</li>"
"<li><b>Condensed Nearest Neighbour:</b> Reduce the dataset size by keeping only necessary examples for classification.</li>"
"<li><b>Cluster Centroids:</b> Resampling data using cluster centroids for the majority class.</li>"
"<li><b>SMOTE:</b> Synthetic Minority Over-sampling Technique to generate synthetic examples for the minority class.</li>"
"<li><b>ADASYN:</b> Adaptive Synthetic Sampling to generate synthetic examples focusing on difficult-to-learn minority examples.</li>"
"<li><b>BorderlineSMOTE:</b> Generating synthetic examples for minority class instances at the decision boundary.</li>"
"</ul>"
"<p>Use the menu on the left to navigate between different features.</p>"


        )
        self.intro_label = QLabel(intro_text)
        self.intro_label.setProperty("class", "heading")
        self.intro_label.setWordWrap(True)
        home_layout.addWidget(self.intro_label, alignment=Qt.AlignmentFlag.AlignTop)

        self.options_frame = QFrame()
        options_layout = QVBoxLayout(self.options_frame)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(10)
        
        self.model_label = QLabel("Select Model:")
        options_layout.addWidget(self.model_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Decision Tree", "SVM", "KNN"])
        self.model_combo.setStyleSheet("""
            QComboBox {
                background-color: #fff;
                border: 1px solid #bbb;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        options_layout.addWidget(self.model_combo)
        
        self.balancing_label = QLabel("Select Balancing Technique:")
        options_layout.addWidget(self.balancing_label)
        self.balancing_combo = QComboBox()
        self.balancing_combo.addItems(["None", "Random Undersampling", "NearMiss", "Tomek Links", "Condensed Nearest Neighbour", "Cluster Centroids", "Random OverSampler", "SMOTE", "ADASYN", "BorderlineSMOTE"])
        self.balancing_combo.setStyleSheet("""
            QComboBox {
                background-color: #fff;
                border: 1px solid #bbb;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        options_layout.addWidget(self.balancing_combo)
        
        self.submit_button = QPushButton("Submit")
        options_layout.addWidget(self.submit_button)
        self.submit_button.setFixedWidth(80)
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        self.submit_button.clicked.connect(self.run_model)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        options_layout.addWidget(self.results_text)

        self.roc_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        options_layout.addWidget(self.roc_canvas)
        self.confusion_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        options_layout.addWidget(self.confusion_canvas)

        home_layout.addWidget(self.options_frame, alignment=Qt.AlignmentFlag.AlignTop)
        self.options_frame.hide()

        self.content_layout.addWidget(home_content)

        # Visualization pages
        self.histogram_page = VisualizationWidget(self)
        self.boxplot_page = VisualizationWidget(self)
        self.heatmap_page = VisualizationWidget(self)
        self.scatter_page = VisualizationWidget(self)
        
        self.content_layout.addWidget(self.histogram_page)
        self.content_layout.addWidget(self.boxplot_page)
        self.content_layout.addWidget(self.heatmap_page)
        self.content_layout.addWidget(self.scatter_page)

        # Data view page
        self.data_view_page = QWidget()
        self.data_view_layout = QVBoxLayout(self.data_view_page)
        self.data_table = QTableWidget()
        self.data_view_layout.addWidget(self.data_table)
        self.content_layout.addWidget(self.data_view_page)

        # ROC Curve page
        self.roc_curve_page = VisualizationWidget(self)
        self.content_layout.addWidget(self.roc_curve_page)

        # Confusion Matrix page
        self.confusion_matrix_page = VisualizationWidget(self)
        self.content_layout.addWidget(self.confusion_matrix_page)

        # Description page
        self.description_page = QWidget()
        self.description_layout = QVBoxLayout(self.description_page)
        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_layout.addWidget(self.description_text)
        self.content_layout.addWidget(self.description_page)
     
        csv_path = os.path.join(application_path, 'creditcard.csv')

        # Load the data
        self.data = pd.read_csv(csv_path)
        # Visualization options page
        self.visualization_options_page = QWidget()
        self.visualization_options_layout = QVBoxLayout(self.visualization_options_page)
        self.visualization_options_layout.setContentsMargins(20, 20, 20, 20)
        self.visualization_options_layout.setSpacing(15)

        self.visualization_options_label = QLabel("Select Visualization Type:")
        self.visualization_options_layout.addWidget(self.visualization_options_label)
        
        self.histogram_button = QPushButton("Histograms")
        self.histogram_button.clicked.connect(self.show_histogram_selection)
        self.visualization_options_layout.addWidget(self.histogram_button)
        
        self.boxplot_button = QPushButton("Boxplots")
        self.boxplot_button.clicked.connect(self.show_boxplot_selection)
        self.visualization_options_layout.addWidget(self.boxplot_button)
        
        self.heatmap_button = QPushButton("Heatmap")
        self.heatmap_button.clicked.connect(self.plot_heatmap)
        self.visualization_options_layout.addWidget(self.heatmap_button)
        
        self.scatter_button = QPushButton("Scatter Plots")
        self.scatter_button.clicked.connect(self.show_scatter_selection)
        self.visualization_options_layout.addWidget(self.scatter_button)
        
        self.visualization_options_page.setLayout(self.visualization_options_layout)
        self.content_layout.addWidget(self.visualization_options_page)

        # Feature selection page
        self.feature_selection_page = QWidget()
        self.feature_selection_layout = QVBoxLayout(self.feature_selection_page)
        self.feature_selection_layout.setContentsMargins(20, 20, 20, 20)
        self.feature_selection_layout.setSpacing(15)

        self.feature_selection_label = QLabel("Select Features:")
        self.feature_selection_layout.addWidget(self.feature_selection_label)

        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.feature_selection_layout.addWidget(self.feature_list)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.plot_selected_visualization)
        self.feature_selection_layout.addWidget(self.next_button)
        
        self.feature_selection_page.setLayout(self.feature_selection_layout)
        self.content_layout.addWidget(self.feature_selection_page)

    def show_home(self):
        self.content_layout.setCurrentIndex(0)
        self.options_frame.hide()
        self.intro_label.show()

    def show_training_options(self):
        self.content_layout.setCurrentIndex(0)
        self.intro_label.hide()
        self.options_frame.show()

    def show_visualization_options(self):
        self.content_layout.setCurrentIndex(self.content_layout.indexOf(self.visualization_options_page))
        self.populate_feature_list()

    def show_histogram_selection(self):
        self.current_visualization = "histogram"
        self.content_layout.setCurrentIndex(self.content_layout.indexOf(self.feature_selection_page))

    def show_boxplot_selection(self):
        self.current_visualization = "boxplot"
        self.content_layout.setCurrentIndex(self.content_layout.indexOf(self.feature_selection_page))

    def show_scatter_selection(self):
        self.current_visualization = "scatter"
        self.content_layout.setCurrentIndex(self.content_layout.indexOf(self.feature_selection_page))

    def populate_feature_list(self):
        self.feature_list.clear()
        for col in self.data.columns:
            self.feature_list.addItem(col)

    def plot_selected_visualization(self):
        selected_features = [item.text() for item in self.feature_list.selectedItems()]
        if self.current_visualization == "histogram":
            self.histogram_page.plot_histograms(self.data[selected_features])
            self.content_layout.setCurrentIndex(self.content_layout.indexOf(self.histogram_page))
        elif self.current_visualization == "boxplot":
            self.boxplot_page.plot_boxplots(self.data[selected_features])
            self.content_layout.setCurrentIndex(self.content_layout.indexOf(self.boxplot_page))
        elif self.current_visualization == "scatter":
            self.scatter_page.plot_scatter(self.data, selected_features)
            self.content_layout.setCurrentIndex(self.content_layout.indexOf(self.scatter_page))

    def plot_heatmap(self):
        self.heatmap_page.plot_heatmap(self.data)
        self.content_layout.setCurrentIndex(self.content_layout.indexOf(self.heatmap_page))

    def view_data(self):
        df = self.data.head(50)
        self.data_table.setRowCount(df.shape[0])
        self.data_table.setColumnCount(df.shape[1])
        self.data_table.setHorizontalHeaderLabels(df.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.data_table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
        self.content_layout.setCurrentIndex(self.content_layout.indexOf(self.data_view_page))

    def run_model(self):
        model_name = self.model_combo.currentText()
        balance_technique = self.balancing_combo.currentText()
        
        # Select the model function based on the user's choice
        if model_name == "Decision Tree":
            model_func = DT_modele
        elif model_name == "SVM":
            model_func = SVM_modele
        elif model_name == "KNN":
            model_func = KNN_modele
        
        data = self.data
        
        # Specify the feature columns and target column
        feature_columns = data.columns[:-1]  # Toutes les colonnes sauf la dernière
        target_column = data.columns[-1]  # La colonne cible
        
        X = data[feature_columns]
        y = data[target_column]
        
        # Optionally sample the data for faster testing
        data_sample = data.sample(frac=0.1, random_state=42)
        X_sample = data_sample[feature_columns]
        y_sample = data_sample[target_column]

        # Create and start the model training thread
        self.thread = ModelTrainingThread(X_sample, y_sample, model_func, balance_technique)
        self.thread.training_completed.connect(self.display_results)
        self.thread.roc_data.connect(self.display_roc_curve)
        self.thread.confusion_matrix_data.connect(self.display_confusion_matrix)
        self.thread.start()

    def display_results(self, scores):
        # Convert scores DataFrame to HTML for display
        scores_html = scores.to_html()
        self.results_text.setHtml(scores_html)

    def display_roc_curve(self, fpr, tpr, auc):
        ax = self.roc_canvas.figure.subplots()
        ax.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc:.2f})')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        self.roc_canvas.draw()

    def display_confusion_matrix(self, cm):
        ax = self.confusion_canvas.figure.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        self.confusion_canvas.draw()

    def show_description(self):
        description_text = (
            "<h3>Data description :</h3>"
            "<p>Info :</p>"
        )
        
        info_buf = StringIO()
        self.data.info(buf=info_buf)
        info_str = info_buf.getvalue().replace('\n', '<br>')
        
        description_text += f"<p>{info_str}</p>"
        
        description_text += "<h3>Descriptive Statistics :</h3>"
        description_text += self.data.describe().to_html()
        
        self.description_text.setHtml(description_text)
        self.content_layout.setCurrentIndex(self.content_layout.indexOf(self.description_page))

class VisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # Store the reference to the main window
        self.layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(14, 10))  # Create the figure with the desired size
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Précédent")
        self.next_button = QPushButton("Suivant")
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        self.layout.addLayout(nav_layout)

    def plot_histograms(self, df):
        self.figure.clear()
        num_rows = (len(df.columns) + 4) // 5
        num_cols = 5
        summary_stats = df.describe()
        axes = self.figure.subplots(num_rows, num_cols, gridspec_kw={'wspace': 0.5, 'hspace': 0.5}).flatten()
        for i, col in enumerate(df.columns):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            sns.kdeplot(df[col].dropna(), ax=axes[i], color='red')
            axes[i].set_title(f'{col}', fontsize=10)
            axes[i].set_xlabel('Value', fontsize=8)
            axes[i].set_ylabel('Frequency', fontsize=8)
            stats_text = (f"Mean: {summary_stats[col]['mean']:.2f}\n"
                          f"Std Dev: {summary_stats[col]['std']:.2f}\n"
                          f"Min: {summary_stats[col]['min']:.2f}\n"
                          f"Max: {summary_stats[col]['max']:.2f}")
            axes[i].annotate(stats_text, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=6, color='black')
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        self.canvas.draw()

    def plot_boxplots(self, df):
        self.figure.clear()
        num_rows = (len(df.columns) + 4) // 5
        num_cols = 5
        summary_stats = df.describe()
        axes = self.figure.subplots(num_rows, num_cols, gridspec_kw={'wspace': 0.5, 'hspace': 0.5}).flatten()
        for i, col in enumerate(df.columns):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(f'{col}', fontsize=10)
            axes[i].set_xlabel('Value', fontsize=8)
            axes[i].set_ylabel('Frequency', fontsize=8)
            stats_text = (f"Mean: {summary_stats[col]['mean']:.2f}\n"
                          f"Std Dev: {summary_stats[col]['std']:.2f}\n"
                          f"Min: {summary_stats[col]['min']:.2f}\n"
                          f"Max: {summary_stats[col]['max']:.2f}")
            axes[i].annotate(stats_text, xy=(0.05, 0.75), xycoords='axes fraction', fontsize=6, color='black')
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        self.canvas.draw()

    def plot_heatmap(self, df):
        self.figure.clear()
        corr = df.corr()
        ax = self.figure.add_subplot(111)
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=plt.cm.Blues, annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        self.canvas.draw()

    def plot_scatter(self, df, features):
        if len(features) != 2:
            print("Please select exactly 2 features for scatter plot")
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(df[features[0]], df[features[1]])
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_title(f'Scatter plot of {features[0]} vs {features[1]}')
        self.canvas.draw()

    def plot_roc_curve(self, fpr, tpr, auc):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc:.2f})')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        self.canvas.draw()

    def plot_confusion_matrix(self, y_test, y_pred):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        self.canvas.draw()

    def show_previous(self):
        self.main_window.content_layout.setCurrentIndex(self.main_window.content_layout.indexOf(self.main_window.feature_selection_page))

    def show_next(self):
        self.main_window.content_layout.setCurrentIndex(self.main_window.content_layout.indexOf(self.main_window.visualization_options_page))
class WelcomePage(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        self.setWindowIcon(QIcon(icon_path)) # Set the icon for the welcome page
        self.setWindowTitle("Welcome to Fraud Detection Project")  # Set the title for the welcome page
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Add background image
        image_label = QLabel(self)
        pixmap = QPixmap(image_path)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(image_label)
        image_label.setScaledContents(True) 
        # Add switch button
        button = QPushButton("Enter", self)
        button.clicked.connect(self.switch_callback)
        button.setFixedSize(100, 40)
        button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

class MainApp(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(icon_path))  # Set the icon for the main app window
        self.setWindowTitle("Machine Learning Interface")  # Set the title for the main app window
        self.welcome_page = WelcomePage(self.show_main_window)
        self.main_window = Window()
        self.addWidget(self.welcome_page)
        self.addWidget(self.main_window)
        self.setCurrentWidget(self.welcome_page)

    def show_main_window(self):
        self.setCurrentWidget(self.main_window)

app = QApplication(sys.argv)
main_app = MainApp()



# Load the Dracula QSS file
with open(qss_path, 'r') as file:
    qss = file.read()
    app.setStyleSheet(qss)

main_app.show()
sys.exit(app.exec())