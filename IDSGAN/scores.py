from sklearn import metrics
from tabulate import tabulate

def get_binary_class_scores(labels, predictions):
    accuracy = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions, zero_division=1)
    precision = metrics.precision_score(labels, predictions, zero_division=1)
    recall = metrics.recall_score(labels, predictions, zero_division=1)
    detection_rate = get_detection_rate(labels, predictions)
    return accuracy, f1, precision, recall, detection_rate

def print_scores(scores):
    scores = list(map(lambda score: f'{score:0.4f}', scores))
    headers = ['accuracy', 'f1', 'precision', 'recall', 'detection_rate']
    print(tabulate([scores], headers=headers))

def get_detection_rate(labels, predictions):
    prediction_mask = [p == 1 for p in predictions]  # Tạo danh sách boolean từ predictions
    if not any(prediction_mask):
        return 0
    number_correctly_detected_attacks = sum(labels[i] for i in range(len(labels)) if prediction_mask[i])
    number_total_attacks = sum(labels)
    if number_total_attacks == 0:
        return 0
    return number_correctly_detected_attacks / number_total_attacks
