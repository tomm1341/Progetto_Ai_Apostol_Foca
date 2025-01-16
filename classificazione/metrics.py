import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

class Metrics:
    
    def __init__(self, classes: list[str], real_y: np.array, pred_y: np.array) -> None:
        self.classes = classes
        self.num_classes = len(classes)
        self.real_y = real_y
        self.pred_y = pred_y
        self.confusion_matrix = None
    
    # Calcola la matrice di confusione fra le classi.
    def compute_confusion_matrix(self) -> None:
        N = self.num_classes
        self.confusion_matrix = np.zeros((N, N), dtype=int)    
        for real, pred in zip(self.real_y, self.pred_y):
            self.confusion_matrix[real][pred] += 1
    
    # Calcola l'accuracy sulla confusion matrix.
    def accuracy(self) -> float:
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
        return np.sum(self.confusion_matrix.diagonal()) / np.sum(self.confusion_matrix)
    
    # Calcola la recall per la classe indicata.
    def recall(self, class_id: int) -> float:
        if not self.__valid_class_id(class_id):
            print(f'Id classe {class_id} invalido.')
            sys.exit(-1)
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
        real_and_predicted = self.confusion_matrix[class_id, class_id]
        real = np.sum(self.confusion_matrix[class_id, :])           
        return 0.0 if real == 0 else (real_and_predicted / real)
    
    # Calcola la precision per la classe indicata.
    def precision(self, class_id: int) -> float:
        if not self.__valid_class_id(class_id):
            print(f'Id classe {class_id} invalido.')
            sys.exit(-1)
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
        real_and_predicted = self.confusion_matrix[class_id, class_id]
        predicted = np.sum(self.confusion_matrix[:, class_id])
        return 0.0 if predicted == 0 else (real_and_predicted / predicted)
    
    # Calcola l'f1-score per la classe indicata.
    def f1_score(self, class_id: int) -> float:
        p, r = self.precision(class_id), self.recall(class_id)
        return 0.0 if (p + r) == 0 else 2 * (p * r) / (p + r)
    
    # Calcola il numero di campioni veri per la classe indicata.
    def support(self, class_id: int) -> int:
        if not self.__valid_class_id(class_id):
            print(f'Id classe {class_id} invalido.')
            sys.exit(-1)
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
        return np.sum(self.confusion_matrix[class_id, :])
    
    # Mostra un report di tutte le informazioni.
    def report(self) -> None:
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
        
        print('')
        print(f'Confusion matrix (accuracy {self.accuracy() * 100:.2f}%):\n\n{self.confusion_matrix}')        
        print(f'\nClass\t\tPrecision\tRecall\tf1-score\tsupport')
        for i, c in enumerate(self.classes):
            print(f'[{c}]\t{self.precision(i):.2f}\t\t{self.recall(i):.2f}\t{self.f1_score(i):.2f}\t\t{self.support(i):.2f}')
        print('')

    # Verifica se l'indice classe e' valido.
    def __valid_class_id(self, class_id: int) -> bool:
        return 0 <= class_id < len(self.classes)
    
    # Getter per ottenere le etichette reali
    def get_real_labels(self):
        return self.real_y
    
    # Getter per ottenere le etichette predette
    def get_predicted_labels(self):
        return self.pred_y


# Funzione per visualizzare la matrice di confusione
def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# Esempio di utilizzo:
if __name__ == '__main__':
    
    real_y = np.random.randint(0, 3, 20)  
    pred_y = np.random.randint(0, 3, 20)

    # Passa le classi in ordine corretto come nel dataset
    mt = Metrics(['bikes', 'cars', 'motorcycles'], real_y, pred_y)
    
    # Mostra il report
    mt.report()

    # Ottenere le etichette reali e predette
    real_labels = mt.get_real_labels()
    predicted_labels = mt.get_predicted_labels()

    print(f'Real labels: {real_labels}')
    print(f'Predicted labels: {predicted_labels}')

    # Visualizza la matrice di confusione
    plot_confusion_matrix(mt.confusion_matrix, mt.classes)





