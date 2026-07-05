import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import sklearn.ensemble
import sklearn.svm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



def seek_best_clasifier(train_metafeaures, test_metafeatures, test_labels, nfew = 10):

    results = []

    metaf_original_test = test_metafeatures
    metaf_original_train = train_metafeaures
    label_original_test = test_labels

    label_original_train = np.zeros(metaf_original_train.shape[0])
    
    # exxtract nfew samples for cross-validation
    nfew_negative = nfew * 10 # from training datase
    nfew_positive = nfew      # from test dataset

    # pickup positives from test dataset
    indices_positive = [i for i in range(len(label_original_test)) if label_original_test[i]==1]
    indices_positive_fewshot = random.sample(indices_positive, nfew_positive) # extract nfew positives
    indices_test_rest = [i for i in range(len(label_original_test)) if i not in indices_positive_fewshot]
    
    #pickp negatives from train dataset
    indices_negative = [i for i in range(len(label_original_train)) if label_original_train[i]==0]
    indices_negative_fewshot = random.sample(indices_negative, nfew_negative) # extract nfew negatives
    indices_train_rest = [i for i in range(len(label_original_train)) if i not in indices_negative_fewshot]    

    # compose
    metaf_training = metaf_original_train[indices_train_rest,:]
    label_training = label_original_train[indices_train_rest]

    metaf_validation = np.concatenate((
        metaf_original_test[indices_positive_fewshot, :],
        metaf_original_train[indices_negative_fewshot, :]
    ), axis=0)
    label_validation = np.concatenate((
        label_original_test[indices_positive_fewshot],
        label_original_train[indices_negative_fewshot]
    ), axis=0)
    
    metaf_test = metaf_original_test[indices_test_rest, :]
    label_test = label_original_test[indices_test_rest   ]
    
    ##
    # one-class classifiers
    
    # each features
    results.append({
        'name': 'recon_loss', 
        'validation': roc_auc_score(label_validation, metaf_validation[:,0]), 
        'test': roc_auc_score(label_test, metaf_test[:,0])
    })

    results.append({
        'name': 'percetual_loss', 
        'validation': roc_auc_score(label_validation, metaf_validation[:,1]), 
        'test': roc_auc_score(label_test, metaf_test[:,1])
    })

    results.append({
        'name': 'range_compression_length', 
        'validation': roc_auc_score(label_validation, metaf_validation[:,2]), 
        'test': roc_auc_score(label_test, metaf_test[:,2])
    })

    # Maharanobis
    mcd = MinCovDet()
    mcd.fit(metaf_original_train)

    results.append({
        'name': 'maharanobis_dist', 
        'validation': roc_auc_score(label_validation, mcd.mahalanobis(metaf_validation)), 
        'test': roc_auc_score(label_test, mcd.mahalanobis(metaf_test))
    })

    # ocsvm
    for nu in (0.01, 0.03, 0.1, 0.3):
        ocsvm = sklearn.svm.OneClassSVM(nu=nu)
        pipe = make_pipeline(StandardScaler(), ocsvm)

        pipe.fit(metaf_original_train)

        results.append({
            'name': f'oneclass_svm_nu%f'%nu, 
            'validation': roc_auc_score(label_validation, -pipe.decision_function(metaf_validation)),
            'test': roc_auc_score(label_test, -pipe.decision_function(metaf_test))
        })

    # isolation forest
    isof = sklearn.ensemble.IsolationForest()

    isof.fit(metaf_validation)

    results.append({
        'name': 'isolation_forest', 
        'validation': roc_auc_score(label_validation, -isof.decision_function(metaf_validation)),
        'test': roc_auc_score(label_test, -isof.decision_function(metaf_test))
    })

    ####
    # learnable (with n-fold CV) 
    
    skf = StratifiedKFold(np.minimum(5, nfew_positive), random_state=42)
    
    #Maharanobis ratio
    assert mcd is not None
    scores = []
    for train_index, test_index in skf.split(metaf_validation, label_validation):
        mcd_trial = EmpiricalCovariance()
        mcd_trial.fit(metaf_validation[train_index], label_validation[train_index])
        score = roc_auc_score(label_validation[test_index], mcd.mahalanobis(metaf_validation[test_index,:]) / mcd_trial.mahalanobis(metaf_validation[test_index,:]))
        scores.append(score)
    
    mcd_trial = EmpiricalCovariance()
    mcd_trial.fit(metaf_validation, label_validation)
    maha_dist_ratio = mcd.mahalanobis(metaf_test) / mcd_trial.mahalanobis(metaf_test)
    
    results.append({
        'name': 'maharanobis_ratio', 
        'validation': np.mean(np.array(scores)), 
        'test': roc_auc_score(label_test, maha_dist_ratio)
    })

    # svm
    scores = []
    for train_index, test_index in skf.split(metaf_validation, label_validation):
        svm_trial = sklearn.svm.SVC()
        pipe = make_pipeline(StandardScaler(), svm_trial)
        pipe.fit(metaf_validation[train_index], label_validation[train_index])
        score = roc_auc_score(label_validation[test_index], pipe.decision_function(metaf_validation[test_index,:]))
        scores.append(score)
    
    svm = sklearn.svm.SVC()
    pipe = make_pipeline(StandardScaler(), svm)
    pipe.fit(metaf_validation, label_validation)
    svm_decision = pipe.decision_function(metaf_test)
    
    results.append({
        'name': 'svm', 
        'validation': np.mean(np.array(scores)), 
        'test': roc_auc_score(label_test, svm_decision)
    })

    # svm with balancing (training data is also used)
    for C in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e-0, 1.0e+1, 1.0e+2, 1.0e+3, 1.0e+4):
        scores = []
        for train_index, test_index in skf.split(metaf_validation, label_validation):
            svm_trial = sklearn.svm.SVC(class_weight='balanced', C=C)
            pipe = make_pipeline(StandardScaler(), svm_trial)
            pipe.fit(
                np.concatenate((metaf_validation[train_index, :], metaf_original_train), axis=0),
                np.concatenate((label_validation[train_index   ], label_original_train), axis=0)
            )
            score = roc_auc_score(label_validation[test_index], pipe.decision_function(metaf_validation[test_index,:]))
            scores.append(score)

        svm = sklearn.svm.SVC(class_weight='balanced', C=C)
        pipe = make_pipeline(StandardScaler(), svm)
        pipe.fit(metaf_validation, label_validation)
        svm_decision = pipe.decision_function(metaf_test)

        results.append({
            'name': f'svm_balancing_C%f'%C, 
            'validation': np.mean(np.array(scores)), 
            'test': roc_auc_score(label_test, svm_decision)
        })

    # random forest
    scores = []
    for train_index, test_index in skf.split(metaf_validation, label_validation):
        rf_trial = sklearn.ensemble.RandomForestClassifier()
        rf_trial.fit(
            np.concatenate((metaf_validation[train_index, :], metaf_original_train), axis=0),
            np.concatenate((label_validation[train_index   ], label_original_train), axis=0)
        )
        score = roc_auc_score(label_validation[test_index], rf_trial.predict_proba(metaf_validation[test_index,:])[:,1])
        scores.append(score)
    
    rf = sklearn.ensemble.RandomForestClassifier()
    rf.fit(metaf_validation, label_validation)
    rf_proba = rf.predict_proba(metaf_test)[:,1]

    results.append({
        'name': 'random_forest', 
        'validation': np.mean(np.array(scores)), 
        'test': roc_auc_score(label_test, rf_proba)
    })



    # select the best classifier
    df = pd.DataFrame.from_records(results)
    
    bestindex = np.argmax(df["validation"].values)
    bestscore = df["test"][bestindex]
    bestmethod = df["name"][bestindex]
    
    print(f"best = %08.8f (%s)" % (bestscore, bestmethod))
    
    print(df)

    return bestscore, bestindex
