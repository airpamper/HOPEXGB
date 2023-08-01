import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import svds, inv
from preprocess import *
import pandas as pd

class HOPE():
    def __init__(self, embedding_dim, beta):
        self.beta = beta
        self.embedding_dim = embedding_dim

    def svd(self, matrix):
        adj = matrix[:, :]

        # compute katz index
        I = eye(m=adj.shape[0], format='csc')
        for i in range(len(adj)):
            adj[i, i] = 0
        adj = csc_matrix(adj)
        S = inv(I - self.beta * adj) * self.beta * adj

        # svd
        u, s, vt = svds(S, k=self.embedding_dim // 2)
        Us = u * (s ** 0.5)
        Vs = vt.T * (s ** 0.5)
        return np.concatenate([Us, Vs], axis=1)


def get_pair_feature(pairs, a_feature, b_feature):
    pair_feature = []
    for pair in pairs:
        a_vec = a_feature[pair[1]]
        b_vec = b_feature[pair[0]]
        pair_vec = np.concatenate([b_vec, a_vec], axis=0).reshape((1, -1)).astype(np.float32)
        pair_feature.append(pair_vec)
    pair_feature = np.concatenate(pair_feature, axis=0)
    return pair_feature



def NegativeSamples(p_feature,n_feature):
    all_dis_ed = []
    n_feature = np.array(n_feature)
    for i in range(0, len(n_feature)):
        dis_ed = []
        for j in range(0, len(p_feature)):
            ed = np.sqrt(np.sum((n_feature.iloc[i,:] - p_feature.iloc[j,:]) ** 2))
            dis_ed.append(ed)
        all_dis_ed.append(dis_ed)

    all_dis_ed = np.array(all_dis_ed)
    k = np.insert(n_feature,0,all_dis_ed,axis=1)
    k_ascend = k[np.argsort(k[:,0])]
    Nsim_n_feature = k_ascend[:,len(p_feature)]
    return Nsim_n_feature
    
def XGB(data,n):
    from sklearn.model_selection import KFold,cross_val_predict
    import xgboost as xgb
    from sklearn import metrics
    x = np.array(data.iloc[:, 1:], dtype=np.float)
    y = np.array(data.iloc[:, 0], dtype=np.float)
    
    assessment_MCC = []
    assessment_AUC = []
    assessment_Accuracy = []
    assessment_Sensitivity = []
    assessment_Specificity = []
    assessment_AUPR = []

    for r in range(n):
        kfold = KFold(n_splits=10,shuffle=True, random_state=r)
        model =xgb.XGBClassifier()
        y_predict = cross_val_predict(model, x, y, cv=kfold)  
        tn, fp, fn, tp = metrics.confusion_matrix(y, y_predict).ravel()
        assessment_AUC.append(metrics.roc_auc_score(y, y_predict))
        assessment_AUPR.append(metrics.average_precision_score(y, y_predict))
        assessment_MCC.append(metrics.matthews_corrcoef(y, y_predict))
        assessment_Accuracy.append(metrics.accuracy_score(y, y_predict))
        assessment_Sensitivity.append(metrics.recall_score(y, y_predict))       
        assessment_Specificity.append(tn / (fn + tn))

    assessment_MCC = np.array(assessment_MCC)
    assessment_AUC = np.array(assessment_AUC)
    assessment_Accuracy = np.array(assessment_Accuracy)
    assessment_Sensitivity = np.array(assessment_Sensitivity)
    assessment_Specificity = np.array(assessment_Specificity)
    assessment_AUPR = np.array(assessment_AUPR)
    
    return_assessment = pd.DataFrame((assessment_MCC.mean().round(4), assessment_MCC.std().round(4), \
                                      assessment_AUC.mean().round(4), assessment_AUC.std().round(4), \
                                      assessment_Accuracy.mean().round(4), assessment_Accuracy.std().round(4), \
                                      assessment_Sensitivity.mean().round(4), assessment_Sensitivity.std().round(4), \
                                      assessment_Specificity.mean().round(4), assessment_Specificity.std().round(4), \
                                      assessment_AUPR.mean().round(4), assessment_AUPR.std().round(4)), \
                                     index=['Mean.MCC', 'SD.MCC', 'Mean.AUC', 'SD.AUC', 'Mean.Accuracy', 'SD.Accuracy', \
                                            'Mean.Sensitivity', 'SD.Sensitivity', 'Mean.Specificity', 'SD.Specificity', \
                                            'Mean.AUPR', 'SD.AUPR'])
    a = return_assessment.T
    return a    
    
if __name__ == "__main__":
    embeding = HOPE(100, 0.01)
    matrix = np.load("all_feature_adj.npy")
    feature = embeding.svd(matrix)
    mi_feature = feature[:578]
    dis_feature = feature[578:958]
    lnc_feature = feature[958:]

    p_feature_mi = get_pair_feature(dis_mi_link, dis_feature, mi_feature)
    p_feature_lnc = get_pair_feature(dis_lnc_link,dis_feature,lnc_feature)
    n_feature_all = get_pair_feature(pairs=unlabeled, dis_feature=dis_feature, mi_feature=mi_feature)    
    p_feature = np.concatenate([p_feature_mi, p_feature_lnc], axis=0)
    p_feature['label'] = 1
    n_feature = NegativeSamples(p_feature,n_feature_all)
    n_feature['label'] = 0
    data = pd.concat([pd.DataFrame(p_feature),pd.DataFrame(n_feature)],axis=0)
    model = XGB()
    results = model(data,10)
