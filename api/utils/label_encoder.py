
class CustomLabelEncoder:
    def __init__(self):
        self.label_to_int = {}
        self.int_to_label = {}

    def fit(self, labels):
        unique_labels = set(labels)
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        self.classes_ = list(self.label_to_int.keys())  # Storing the class names

    def transform(self, labels):
        return [self.label_to_int.get(label, -1) for label in labels]

    def inverse_transform(self, ints):
        transformed_labels = [self.int_to_label.get(str(i), "Unknown") for i in ints]
        return transformed_labels

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)

    def set_mappings(self, label_to_int, int_to_label):
        self.label_to_int = label_to_int
        self.int_to_label = int_to_label
        self.classes_ = [int_to_label[str(i)] for i in range(len(int_to_label))]  # Ensuring correct order

