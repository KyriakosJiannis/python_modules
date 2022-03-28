""" Custom metrics for FLAML
 flaml supports:
 'accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'f1', 'micro_f1', 'macro_f1', 'log_loss', 'mae',
 'mse', 'r2', 'mape'.
 ref: https://github.com/microsoft/FLAML/blob/main/flaml/automl.py
"""


def flaml_recall(X_val, y_val, estimator, labels, X_train, y_train,
                 weight_val=None, weight_train=None, config=None,
                 groups_val=None, groups_train=None):
    from sklearn.metrics import recall_score
    import time
    start = time.time()

    y_pred = estimator.predict(X_val)

    pred_time = (time.time() - start) / len(X_val)

    val_loss = recall_score(y_val, y_pred, labels=labels, sample_weight=weight_val)
    y_pred = estimator.predict(X_train)

    train_loss = recall_score(y_train, y_pred, labels=labels, sample_weight=weight_train)

    return 1 - val_loss, {"val_loss": 1 - val_loss, "train_loss": train_loss, "pred_time": pred_time}
