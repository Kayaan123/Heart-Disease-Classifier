from dataclasses import dataclass
import pandas as pd 
from typing import List, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, log_loss, brier_score_loss, f1_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
data_path="heart.csv"

class MixedNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, var_smoothing=1e-9, alpha: float = 1.0, priors: Optional[np.ndarray] = None, min_categories: Optional[List[int]] = None):
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.priors = priors
        self.min_categories = min_categories

        self._le = LabelEncoder()
        self._gnb = None
        self._cnb = None
        self.classes_ = None
        self.class_log_prior_ = None
        self._disc_maps: List[dict] = []
        self._disc_K: List[int] = []
    
    #if training data for discrete columns is not in the correct form

    def _fit_disc_encoder(self, X_disc: np.ndarray): 
        
        n_cols = X_disc.shape[1]
        self._disc_maps = []
        self._disc_K = []
        for j in range(n_cols):
            vals = np.unique(X_disc[:, j])
            val_to_code = {v: i for i, v in enumerate(vals)}
            if self.min_categories is not None:
                K = int(self.min_categories[j])
                
            else:
                K = len(vals)
            self._disc_maps.append(val_to_code)
            self._disc_K.append(K)


    #assign value to a discrete column up from 0 - K-1
    def _transform_disc(self, X_disc: np.ndarray) -> np.ndarray:
        X=np.empty_like(X_disc, dtype = np.int64) #empty of size X_disc
        for j, (mp, K) in enumerate(zip(self._disc_maps, self._disc_K)):
            col = X_disc[:,j]
            def map_one(v):
                if v in mp:
                    return mp[v]
                if self.min_categories is not None and K > len(mp):
                    return K - 1
                return mp.get(v,0)
            X[:,j] = np.fromiter((map_one(v) for v in col), dtype = np.int64, count = len(col))
        return X
    
    #create Gaussian and Categorical Naive Bayes models
    def fit(self, X_cont: np.ndarray, X_disc: np.ndarray, y: np.ndarray):
        y_enc = self._le.fit_transform(y)
        self.classes_ = self._le.classes_

        
        if self.priors is None:
            counts = np.bincount(y_enc, minlength=len(self.classes_))
            priors = counts / counts.sum()
        else:
            priors = np.asarray(self.priors, dtype=float)
            priors = priors / priors.sum()

        self.class_log_prior_ = np.log(priors)

        
        self._gnb = GaussianNB(priors=priors, var_smoothing=self.var_smoothing)
        self._gnb.fit(np.asarray(X_cont, dtype=float), y_enc)

        
        self._fit_disc_encoder(np.asarray(X_disc))
        Xd_enc = self._transform_disc(np.asarray(X_disc))
        if self.min_categories is not None:
            cnb = CategoricalNB(alpha=self.alpha, class_prior=priors, min_categories=self.min_categories)
        else:
            cnb = CategoricalNB(alpha=self.alpha, class_prior=priors)
        self._cnb = cnb.fit(Xd_enc, y_enc)

        return self


    #combine the predictions of the 2 models together to get an overall prediction
    def _combined_log(self, X_cont: np.ndarray, X_disc: np.ndarray) -> np.ndarray:
        Lg = self._gnb.predict_log_proba(np.asarray(X_cont, dtype=float))
        Lc = self._cnb.predict_log_proba(self._transform_disc(np.asarray(X_disc)))
        
        L = Lg + Lc - self.class_log_prior_
        return L 

    def predict(self, X_cont: np.ndarray, X_disc: np.ndarray) -> np.ndarray:
        L = self._combined_log(X_cont, X_disc)
        idx = L.argmax(axis=1)
        return self._le.inverse_transform(idx)
    


    def predict_log_proba(self, X_cont: np.ndarray, X_disc: np.ndarray) -> np.ndarray:
        L = self._combined_log(X_cont, X_disc)
        
        return L - logsumexp(L, axis=1, keepdims=True)

    def predict_proba(self, X_cont: np.ndarray, X_disc: np.ndarray) -> np.ndarray:
        logP = self.predict_log_proba(X_cont, X_disc)
        return np.exp(logP)

#extract from data
def extractData(src):
    df = pd.read_csv(src)

    continuousCols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    discreteCols   = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    target_col     = df.columns[-1]   


    y = df[target_col].to_numpy()


    X_cont = df[continuousCols].to_numpy(dtype=float)
    X_disc = df[discreteCols].to_numpy()  

    return X_cont, X_disc, y



#implement a stratified k fold, and for validation sets, log the probability the model predicts and the actual value from the data
#to be used in a calibration process

def oof_probs_for_mixed_nb(xc, xd, y, n_splits=5, random_state=42,
                           var_smoothing=1e-9, alpha=1.0, min_cats=None):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pos = np.zeros(len(y), dtype=float)   

    for tr_idx, va_idx in skf.split(xc, y):
        xc_tr, xc_va = xc[tr_idx], xc[va_idx]
        xd_tr, xd_va = xd[tr_idx], xd[va_idx]
        y_tr,  y_va  = y[tr_idx],  y[va_idx]

        clf = MixedNaiveBayes(var_smoothing=var_smoothing, alpha=alpha, min_categories=min_cats)
        clf.fit(xc_tr, xd_tr, y_tr)

        pos_idx = int(np.where(clf.classes_ == 1)[0][0])    
        oof_pos[va_idx] = clf.predict_proba(xc_va, xd_va)[:, pos_idx]

    return oof_pos


def _logit_clip(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

class PlattCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(solver="lbfgs")

    def fit(self, p_uncal, y):
        z = _logit_clip(p_uncal).reshape(-1, 1)  
        self.lr.fit(z, y)
        return self

    def predict(self, p_uncal):
        z = _logit_clip(p_uncal).reshape(-1, 1)
        return self.lr.predict_proba(z)[:, 1]
def main():
    X_cont, X_disc, y = extractData(data_path)
    xcont_tr, xcont_te, xdisc_tr, xdisc_te, y_tr, y_te = train_test_split(
        X_cont, X_disc, y, test_size=0.2, random_state=42, stratify=y
    )

    min_cats = [2,4,2,3,2,3,4,4]

    
    p_oof = oof_probs_for_mixed_nb(
        xcont_tr, xdisc_tr, y_tr,
        n_splits=5, random_state=42,
        var_smoothing=1e-9, alpha=1.0, min_cats=min_cats
    )
    
    assert p_oof.shape == y_tr.shape

    
    cal = PlattCalibrator().fit(p_oof, y_tr)

    
    clf = MixedNaiveBayes(var_smoothing=1e-9, alpha=1.0, min_categories=min_cats)
    clf.fit(xcont_tr, xdisc_tr, y_tr)

    
    pos_idx = int(np.where(clf.classes_ == 1)[0][0])
    probs_test_uncal = clf.predict_proba(xcont_te, xdisc_te)[:, pos_idx]

    
    probs_test_cal = cal.predict(probs_test_uncal)
    y_pred_05 = (probs_test_cal >= 0.5).astype(int)

    print("ROC AUC (cal):", roc_auc_score(y_te, probs_test_cal))
    print("Log loss (cal):", log_loss(y_te, np.c_[1 - probs_test_cal, probs_test_cal]))
    print("Brier (cal):", brier_score_loss(y_te, probs_test_cal))
    print("Accuracy @0.5:", accuracy_score(y_te, y_pred_05))
    print("F1 @0.5:", f1_score(y_te, y_pred_05))
    print("\nConfusion matrix @0.5:\n", confusion_matrix(y_te, y_pred_05))


if __name__ == "__main__":
    main()




