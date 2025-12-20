import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

class VisualizationUtils:
    """
    Utility class for data visualization and dataset analysis.
    """
    
    @staticmethod
    def plot_class_distribution(y_train, y_test, target_name="Target"):
        """
        Calculates and displays the distribution of classes in train and test sets.
        
        Args:
            y_train: Training labels
            y_test: Testing labels
            target_name (str): Name of the target variable for display
        """
        # Calculation of percentages
        train_counts = pd.Series(y_train).value_counts()
        train_pct = pd.Series(y_train).value_counts(normalize=True) * 100
        
        test_counts = pd.Series(y_test).value_counts()
        test_pct = pd.Series(y_test).value_counts(normalize=True) * 100
        
        # Summary DataFrame
        dist_df = pd.DataFrame({
            'Train Count': train_counts,
            'Train %': train_pct,
            'Test Count': test_counts,
            'Test %': test_pct
        }).sort_index()
        
        print(f"\nClass Distribution Analysis ({target_name})")
        print("=" * 65)
        print(dist_df.to_string(float_format=lambda x: f"{x:.2f}"))
        print("=" * 65)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Palette: Blue for non-hazardous, Red/Orange for hazardous
        palette = {0: "#45b6fe", 1: "#ff6b6b"}
        
        sns.countplot(x=y_train, ax=axes[0], palette=palette, hue=y_train, legend=False)
        axes[0].set_title(f"Train Set (n={len(y_train)})", fontweight='bold')
        axes[0].set_xlabel(f"{target_name} (0: No, 1: Yes)")
        
        sns.countplot(x=y_test, ax=axes[1], palette=palette, hue=y_test, legend=False)
        axes[1].set_title(f"Test Set (n={len(y_test)})", fontweight='bold')
        axes[1].set_xlabel(f"{target_name} (0: No, 1: Yes)")
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_matrix(df, title="Feature Correlation Matrix"):
        """Plots a heatmap of feature correlations."""
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(df.corr(), dtype=bool))
        sns.heatmap(df.corr(), mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title, fontweight='bold')
        plt.show()

    @staticmethod
    def plot_model_performance(model_obj, X_test, y_test):
        """
        Affiche la matrice de confusion et la courbe ROC sur une seule ligne.
        """
        y_pred = model_obj.predict(X_test)
        model_name = model_obj.name
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Évaluation des performances : {model_name}", fontsize=16, fontweight='bold')

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Non-Dangereux', 'Dangereux'],
                    yticklabels=['Non-Dangereux', 'Dangereux'])
        ax1.set_title("Matrice de Confusion")
        ax1.set_xlabel("Prédiction")
        ax1.set_ylabel("Vrai label")

        if hasattr(model_obj.model, "predict_proba"):
            probs = model_obj.model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)

            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_title("Courbe ROC")
            ax2.set_xlabel("Taux de Faux Positifs (FPR)")
            ax2.set_ylabel("Taux de Vrais Positifs (TPR)")
            ax2.legend(loc="lower right")
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "Courbe ROC non disponible\n(Le modèle ne supporte pas predict_proba)", 
                    ha='center', va='center', fontsize=12, color='red')
            ax2.set_title("Courbe ROC (Indisponible)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()