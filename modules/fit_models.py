def fit_model(model, train_data, validation_data, epochs, callbacks, save_path):
    """
    Shows the model architecture, fits the model and saves the model
    INPUT:
        model: Model (keras.engine.sequential.Sequential)
        train_data: Map training dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
        validation_data: Map validation dataset (tensorflow.python.data.ops.dataset_ops.MapDataset)
        epochs: Number of epochs (int)
        callbacks: List of callbacks (list)
        save_path: Path where model .keras is saved (str)
    OUTPUT:
        model: Trained model (keras.engine.sequential.Sequential)
        trace: Trace of trained model where model.fit is stored (keras.callbacks.History)
    """
    
    model.summary()
    trace = model.fit(train_data, validation_data=validation_data, epochs=epochs, callbacks=callbacks)
    model.save(save_path)
    
    return model, trace