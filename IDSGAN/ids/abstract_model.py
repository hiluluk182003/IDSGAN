from joblib import dump, load
import os


class AbstractModel:
    """
    Base model that all other models should inherit from.
    Expects that classifier algorithm is initialized during construction.
    """

    def train(self, X_train, y_train, X_val, y_val):
        self.classifier.fit(X_train, y_train)

    def predict(self, X):
        return self.classifier.predict(X)

    def save(self, path):
    # Kiểm tra nếu thư mục chưa tồn tại, hãy tạo mới nó
     os.makedirs(os.path.dirname(path), exist_ok=True)
     dump(self.classifier, path)

    def load(self, path):
        self.classifier = load(path)
        