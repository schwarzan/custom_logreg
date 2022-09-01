from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class ThresholdedLogRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, LogReg=LogisticRegression(), threshold=0.28):
        self.lr = LogReg
        self.threshold = threshold
        
    def fit(self, X, y=None):
        self.lr.fit(X, y)
    
    def predict(self, X, y=None):
        
        preds = (self.lr.predict_proba(X)[:,1] >= self.threshold).astype(int)       
        return preds
    
    def predict_proba(self, X, y=None):
        
        preds = self.lr.predict_proba(X)
        return preds
