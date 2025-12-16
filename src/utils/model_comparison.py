import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import os
from IPython.display import display, Markdown

from src.models.tree_decision_classifier import DecisionTreeModel
from src.models.svm_classifier import SVMClassifier
from src.models.logistic_regression_classifier import LogisticRegressionClassifier
from src.models.naive_bayes_classifier import NaiveBayesClassifier
from src.models.nn_classifier import NNClassifier
from src.models.least_squares_classifier import LeastSquaresClassifier
from src.config import get_config


class ModelComparator:
    """
    A class to compare multiple classification models.

    Features:
    - Train all models
    - Collect metrics (accuracy, precision, recall, F1, AUC-ROC, time, memory)
    - Display results in a formatted table
    - Plot comparative bar charts
    """

    def __init__(self):
        self.models = {}
        self.results = {}  # Stores mean ± std for each model
        self.config = get_config()

    def add_model(self, model_instance):
        """
        Add a model to the comparison.

        Args:
            model_instance: An instance of a model class (inheriting from BaseModel)
        """
        self.models[model_instance.name] = model_instance

    def add_models_default(self):
        """
        Add all default models with their default configurations.
        """
        default_models = [
            DecisionTreeModel(name="Decision Tree", params={}),
            SVMClassifier(name="SVM", params={}),
            LogisticRegressionClassifier(
                name="Logistic Regression", params={}),
            NaiveBayesClassifier(name="Naive Bayes", params={}),
            NNClassifier(name="Neural Network", params={}),
            LeastSquaresClassifier(name="Least Squares", params={})
        ]

        for model in default_models:
            self.add_model(model)

    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test, n_runs=1):
        """
        Train and evaluate all added models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            n_runs: Number of runs for each model

        Returns:
            dict: Results for all models
        """
        print("STARTING MODEL COMPARISON\n")

        for model_name, model in self.models.items():
            print(f"\n[{model_name}]\n")

            # Temporary storage for runs
            run_metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'auc_roc': [],
                'temps': [],
                'memoire': []
            }

            for run in range(n_runs):
                if n_runs > 1:
                    print(f"  Run {run + 1}/{n_runs}...")

                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024

                start_time = time.time()
                model.train(X_train, y_train)
                train_time = time.time() - start_time

                mem_after = process.memory_info().rss / 1024 / 1024
                mem_used = mem_after - mem_before

                # Evaluate
                metrics = model.evaluate(X_test, y_test)

                # Store metrics for this run
                run_metrics['accuracy'].append(metrics['accuracy'])
                run_metrics['precision'].append(metrics['precision'])
                run_metrics['recall'].append(metrics['recall'])
                run_metrics['f1_score'].append(metrics['f1_score'])

                # Handle AUC-ROC (might be "N/A" for some models)
                if metrics['auc_roc'] != "N/A":
                    run_metrics['auc_roc'].append(metrics['auc_roc'])
                else:
                    run_metrics['auc_roc'].append(np.nan)

                run_metrics['temps'].append(train_time)
                run_metrics['memoire'].append(max(mem_used, 0))

            # Compute mean ± std and store only that
            self.results[model_name] = {
                'accuracy': (np.mean(run_metrics['accuracy']), np.std(run_metrics['accuracy'])),
                'precision': (np.mean(run_metrics['precision']), np.std(run_metrics['precision'])),
                'recall': (np.mean(run_metrics['recall']), np.std(run_metrics['recall'])),
                'f1_score': (np.mean(run_metrics['f1_score']), np.std(run_metrics['f1_score'])),
                'auc_roc': (np.nanmean(run_metrics['auc_roc']), np.nanstd(run_metrics['auc_roc'])),
                'temps': (np.mean(run_metrics['temps']), np.std(run_metrics['temps'])),
                'memoire': (np.mean(run_metrics['memoire']), np.std(run_metrics['memoire'])),
                'confusion_matrix': metrics['confusion_matrix']  # Last run
            }

            print(f"  ✓ {model_name} completed!")

        print("\nCOMPARISON COMPLETED\n")
        return self.results

    def display_results_table(self):
        """
        Display results in a formatted table showing mean ± std for each metric.
        """
        if not self.results:
            print("No results to display. Run train_and_evaluate_all() first.")
            return

        metrics_info = {
            'Accuracy': ('accuracy', '.4f'),
            'Precision': ('precision', '.4f'),
            'Recall': ('recall', '.4f'),
            'F1-Score': ('f1_score', '.4f'),
            'AUC-ROC': ('auc_roc', '.4f'),
            'Temps (s)': ('temps', '.4f'),
            'Mémoire (MB)': ('memoire', '.2f')
        }

        table_data = []
        for model_name, metrics in self.results.items():
            row = {'Méthode': model_name}

            for display_name, (metric_key, fmt) in metrics_info.items():
                mean, std = metrics[metric_key]
                row[display_name] = f"{mean:{fmt}} ± {std:{fmt}}"

            table_data.append(row)

        df_results = pd.DataFrame(table_data)

        print("\nTABLEAU COMPARATIF DES MODÈLES\n")
        print(df_results.to_string(index=False))

        return df_results

    def plot_comparison(self, figsize=None, n_cols=None, save_path=None):
        """
        Plot comparative bar charts for all metrics with error bars.

        Args:
            figsize: Figure size (width, height). If None, uses config value.
            n_cols: Number of columns in the subplot grid. If None, uses config value.
            save_path: If provided, save the figure to this path
        """
        if not self.results:
            print("No results to plot. Run train_and_evaluate_all() first.")
            return

        # Use config defaults if not provided
        if figsize is None:
            figsize = tuple(self.config.get_param(
                'model_comparison.plot_figsize', [20, 12]))
        if n_cols is None:
            n_cols = self.config.get_param('model_comparison.plot_n_cols', 4)

        # Metrics to plot
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score',
                           'AUC-ROC', 'Temps (s)', 'Mémoire (MB)']

        metrics_keys = {
            'Accuracy': 'accuracy',
            'Precision': 'precision',
            'Recall': 'recall',
            'F1-Score': 'f1_score',
            'AUC-ROC': 'auc_roc',
            'Temps (s)': 'temps',
            'Mémoire (MB)': 'memoire'
        }

        n_metrics = len(metrics_to_plot)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        # Color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results)))

        # Plot each metric
        for i, metric_name in enumerate(metrics_to_plot):
            ax = axes[i]
            metric_key = metrics_keys[metric_name]

            # Extract mean and std for all models
            methodes = []
            means = []
            stds = []

            for model_name, metrics in self.results.items():
                mean, std = metrics[metric_key]
                if not (np.isnan(mean) and np.isnan(std)):
                    methodes.append(model_name)
                    means.append(mean)
                    stds.append(std)

            if not methodes:
                ax.set_visible(False)
                continue

            x = np.arange(len(methodes))

            # Create bars
            bars = ax.bar(
                x,
                means,
                yerr=stds,
                capsize=5,
                color=colors[:len(methodes)],
                alpha=0.8,
                edgecolor='black',
                linewidth=1.2
            )

            # Styling
            ax.set_title(f'{metric_name}', fontsize=14, weight='bold', pad=10)
            ax.set_ylabel('Score' if metric_name not in [
                          'Temps (s)', 'Mémoire (MB)'] else metric_name)
            ax.set_xticks(x)
            ax.set_xticklabels(methodes, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Add value labels on bars
            for j, (bar, mean) in enumerate(zip(bars, means)):
                if not np.isnan(mean):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{mean:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )

        # Hide unused subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)

        # Overall title
        fig.suptitle(
            'Comparaison des Modèles de Classification - Métriques de Performance',
            fontsize=18,
            weight='bold',
            y=0.995
        )

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Figure saved to: {save_path}")

        plt.show()

    def get_best_model(self, metric='accuracy'):
        """
        Get the best performing model based on a specific metric.

        Args:
            metric: Metric to use for comparison (default: 'accuracy')

        Returns:
            tuple: (model_name, mean_metric_value)
        """
        if not self.results:
            print("No results available. Run train_and_evaluate_all() first.")
            return None

        best_model = None
        best_score = -np.inf

        for model_name, metrics in self.results.items():
            if metric in metrics:
                # Extract mean from (mean, std) tuple
                mean_score, _ = metrics[metric]
                if not np.isnan(mean_score) and mean_score > best_score:
                    best_score = mean_score
                    best_model = model_name

        return best_model, best_score

    def display_summary(self):
        """Display a summary of the comparison."""
        if not self.results:
            print("No results available. Run train_and_evaluate_all() first.")
            return

        print("SUMMARY - BEST MODELS BY METRIC\n")

        metrics_to_check = ['accuracy', 'precision',
                            'recall', 'f1_score', 'auc_roc']

        for metric in metrics_to_check:
            best_model, best_score = self.get_best_model(metric)
            if best_model:
                print(
                    f"  🏆 Best {metric.upper()}: {best_model} ({best_score:.4f})")

        # Fastest model
        fastest_model = None
        fastest_time = np.inf
        for model_name, metrics in self.results.items():
            # Extract mean from (mean, std) tuple
            mean_time, _ = metrics['temps']
            if mean_time < fastest_time:
                fastest_time = mean_time
                fastest_model = model_name

        print(f"  ⚡ Fastest: {fastest_model} ({fastest_time:.4f}s)")
        print("=" * 60)

    def run_full_comparison(self, df, n_runs=1, save_plot=None, show_plot=True):
        """
        Run a complete comparison pipeline from raw data to results.

        Args:
            df: Raw DataFrame with features and target
            n_runs: Number of runs for each model (default: 1)
            save_plot: Path to save the comparison plot (default: None)
            show_plot: Whether to display the plot (default: True)

        Returns:
            dict: Results for all models
        """
        from src.data.preprocessor import NEODataPreprocessor

        print("MODEL COMPARISON - CLASSIFICATION\n")

        # Preprocessing
        print("\n[1] Preprocessing data...")
        preprocessor = NEODataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
        preprocessor.display_split_info()

        # Add default models if none added
        if not self.models:
            print("\n[2] Adding default models...")
            self.add_models_default()

        # Train and evaluate
        print("\n[3] Training and evaluating models...")
        self.train_and_evaluate_all(
            X_train, y_train, X_test, y_test, n_runs=n_runs)

        # Display results
        print("\n[4] Displaying results...")
        self.display_summary()
        self.display_results_table()

        # Plot comparison
        if show_plot:
            print("\n[5] Generating comparison plots...")
            self.plot_comparison(save_path=save_plot)

        print("\nCOMPARISON COMPLETED!")

        return self.results
