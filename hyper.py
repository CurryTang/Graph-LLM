def hyper_search(trial, embedding_type, gnn_model, dataset):
    """
        an example, set the range here
    """
    return {
                'lr': trial.suggest_categorical('lr', [1e-2, 5e-2, 5e-3, 1e-3]),
                'weight_decay': trial.suggest_categorical('weight_decay', [1e-5, 5e-5, 5e-4, 0]),
                'hidden_dimension': trial.suggest_categorical('hidden_dimension', [16, 32, 64, 8, 128, 256]),
                'dropout': trial.suggest_categorical('dropout', [0., .1, .2, .3, .5, .8]),
                'num_layers': trial.suggest_categorical('num_layers', [2,3]), 
                'normalize_features': trial.suggest_categorical('normalize_features', [0, 1])
            }