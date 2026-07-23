import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_history(trace):
    """
    Plots training and validation loss and accuracy
    INPUT:
        trace: Trace of trained model where model.fit is stored (keras.callbacks.History)
    """

    epochs = np.arange(1, len(trace.history['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6))

    ax1.plot(epochs, trace.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in trace.history:
        ax1.plot(epochs, trace.history['val_loss'], label='Validation Loss', linewidth=2)

    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xticks(epochs)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, trace.history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in trace.history:
        ax2.plot(epochs, trace.history['val_accuracy'], label='Validation Accuracy', linewidth=2)

    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    

def get_model_predictions(model, test_data):
    """
    Extracts test loss, test accuracy, the probability associated with each class for each instance on the test set
    and the digit with the highest probability, which represents the model prediction
    INPUT:
        model: Trained model (keras.engine.sequential.Sequential)
        test_data: Test dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
    OUTPUT:
        test loss: Test loss (float)
        test_accuracy: Test accuracy (float)
        probabilities: Matrix with probabilities of each class for each instance on the test set (numpy.ndarray)
        y_pred: Digit predicted for each instance on the test set (numpy.ndarray)
    """

    test_loss, test_accuracy = model.evaluate(test_data, verbose=0)
    probabilities = model.predict(test_data, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    return test_loss, test_accuracy, probabilities, y_pred


def show_confusion_matrix(y_test, y_pred):
    """
    Shows the confusion matrix that compares the model predictions vs. the actual digits on the test set
    INPUT:
        y_test: Labels of the test dataframe (pandas.core.series.Series)
        y_pred: Digit predicted for each instance on the test set (numpy.ndarray)
    """
    
    plt.figure(figsize=(5,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', cbar=False)
    plt.xlabel('Predicted digit')
    plt.ylabel('True digit')
    plt.show()


def show_classification_report(y_test, y_pred):
    """
    Shows the classification report with precision, recall, f1-score and support for each class
    INPUT:
        y_test: Labels of the test dataframe (pandas.core.series.Series)
        y_pred: Digit predicted for each instance on the test set (numpy.ndarray)
    """
    
    print(classification_report(y_test, y_pred, digits=4))


def plot_top_confident_wrong_predictions(X_test, y_test, probabilities, y_pred, top_k):
    """
    Plots the images that where predicted incorrectly with the highest probability on the wrong digit
    INPUT:
        X_test: Features of the test dataframe (pandas.core.frame.DataFrame)
        y_test: Labels of the test dataframe (pandas.core.series.Series)
        probabilities: Matrix with probabilities of each class for each instance on the test set (numpy.ndarray)
        y_pred: Digit predicted for each instance on the test set (numpy.ndarray)
        top_k: Number of digits to plot (int)
    """
    
    wrong_mask = (y_pred != y_test.to_numpy())
    wrong_indices = np.where(wrong_mask)[0]
    wrong_probs_predclass = probabilities[wrong_indices, y_pred[wrong_indices]]
    order_conf = np.argsort(-wrong_probs_predclass)
    top_wrong_indices = wrong_indices[order_conf[:top_k]]

    ncols = 5
    nrows = int(np.ceil(top_k / ncols))
    
    plt.figure(figsize=(2.2*ncols, 2.3*nrows))
    for i, idx in enumerate(top_wrong_indices):
        img = X_test.to_numpy()[idx].reshape(28, 28)
        t = y_test.to_numpy()[idx]
        p = y_pred[idx]
        pr = probabilities[idx, p]
    
        ax = plt.subplot(nrows, ncols, i+1)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"True:{t}   Predicted:{p}", fontsize=9)
    
    plt.tight_layout()
    plt.show()