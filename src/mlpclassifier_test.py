from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

class MLPClassificationModel:
    def __init__(self, hidden_layer_sizes, learning_rate_init, max_iter):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            solver="sgd",              # <--- HERE
            momentum=0.9,              # <--- HERE
            learning_rate="adaptive",  # <--- HERE
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)


    def predict(self, X_test):
        return self.model.predict(X_test)


    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return y_pred
    
'''
***How to use:

clf = MLPClassificationModel(
    hidden_layer_sizes=(64, 32),
    learning_rate_init=0.001,
    max_iter=300
)

clf.train(X_train_scaled, y_train)
y_pred = clf.evaluate(X_test_scaled, y_test)
'''