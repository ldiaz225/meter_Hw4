import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ConfusionMatrixVisualizer:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred
        self.cm = confusion_matrix(y_test, y_pred)

    def plot_confusion_matrix(self):
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            self.cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted 1", "Predicted 2"],
            yticklabels=["Actual 1", "Actual 2"]
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()

    def plot_performance_scatter(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.6)
        plt.xlabel("Actual Class")
        plt.ylabel("Predicted Class")
        plt.title("MLPClassifier Performance")
        plt.grid(True)
        plt.show()

'''
***How to use:

# Create model with your chosen hyperparameters
clf = MLPClassificationModel(
    hidden_layer_sizes=(64, 32),
    learning_rate_init=0.001,
    max_iter=300
)

# Train
clf.train(X_train_scaled, y_train)

# Evaluate
y_pred = clf.evaluate(X_test_scaled, y_test)

# Visualize
viz = ConfusionMatrixVisualizer(y_test, y_pred)
viz.plot_confusion_matrix()
viz.plot_performance_scatter()
'''