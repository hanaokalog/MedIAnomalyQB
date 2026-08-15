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
import scipy.stats as st



# finding the best fewshot classifier & hyperparameter set
# based on
# Demsar (2006) "Statistical Comparisons of Classifiers over Multiple Data Sets" (JMLR 7:1-30)
# 



class BestModelSelector():
    def __init__(self, n_fewshot : int):
        self.N = n_fewshot
        self.data = []
        self.best_test_score = -np.inf
    
    def add(self, description : str, fewshot_scores : list[float], test_score : float):
        # input:
        #   description     description of the experiment
        #   fewshot_scores  [AUROC with (all negatives + i-th positive) dataset] for all i in 1...N  larger is better
        #   test_score      (secret) score with the test dataset  larger is better
        
        self.data.append({
            "description": description,
            "fewshot_scores": np.array(fewshot_scores),
            "test_score": test_score
        })
        
        self.best_test_score = np.maximum(self.best_test_score, test_score)
    
    def get_current_best(self):
        # calculate ranks
        rank_matrix = np.ndarray(self.N, len(self.data))
        
        for j, d in enumerate(self.data):
            rank_matrix[:, j] = d["fewshot_scores"]
        
        for i in range(self.N):
            rank_matrix[i, :] = np.argsort(np.argsort(rank_matrix[i, :]))
        
        # calculate sum rank
        avg_rank = np.sum(rank_matrix, axis=0)
        
        # best model
        best_j = np.argmin(avg_rank)
        
        # output the best classifier/model with its test-dataset score
        return {
            "best_model_descriptor": self.data[best_j].description,
            "best_avg_rank": np.max(avg_rank),
            "test_score_with_the_best": self.data[best_j].test_score,
            "peeked_current_real_best_test_score": self.best_test_score
        }



class FewshotClassifierTester():
    def __init__(self, 
                 n_fewshot : int, 
                 train_labels : np.array, 
                 test_labels : np.array, 
                 random_seed : int
                 ):
        self.N = n_fewshot
        self.selector = BestModelSelector(self.N)
        
        assert(np.all(train_labels == 0))
        
        random.seed(random_seed)

        # build integrated binary and index set

        # pickup N positives from the test data
        self.indices_positive_in_test = np.where(test_labels!=0)[0]
        self.indices_positive_in_test_partial = np.array(random.sample(list(self.indices_positive_in_test), self.N))

        # pickup half negatives from the train data
        self.indices_negative_in_train = np.where(train_labels==0)[0]
        self.indices_negative_in_train_partial = np.array(random.sample(list(self.indices_negative_in_train), len(list(self.indices_negative_in_train))//2))

        #integrate train and test datasets
        self.n_train = len(train_labels)
        self.n_test = len(test_labels)
        self.n_total = self.n_train + self.n_test
        self.labels = np.concatenate((train_labels, test_labels), axis=0)
        
         # binarize indices
        def binarize(indices):
            barr = np.zeros((self.n_total,), dtype=bool)
            barr[indices] = True
            return barr

        # indexer
        def indexer(binaries):
            return np.where(binaries!=0)[0]

        # negator
        def negate(indices):
            return indexer(~binarize(indices))

        self.bool_train = np.array([0,]*self.n_train + [1,]*self.n_total)
        self.bool_test = ~self.bool_train

       # indexing
        self.indices_fewshot = np.array(self.indices_positive_in_test_partial) + self.n_train
        self.indices_genuine_test = indexer(~binarize(self.indices_fewshot) & self.bool_test)
        self.indices_train1_for_loo_evaluation = self.indices_negative_in_train_partial
        self.indices_train2_for_loo_training = indexer(~binarize(self.indices_trainer_in_train) & self.bool_train)

        # bool arrays
        self.bool_fewshot = binarize(self.indices_fewshot)
        self.bool_genuine_test = binarize(self.genuine_test)
        self.bool_train1_for_loo_evaluation = binarize(self.indices_train1_for_loo_evaluation)
        self.bool_train2_for_loo_training = binarize(self.indices_train2_for_loo_training)

        # final trainable set (not for LOO)
        self.bool_all_train_plus_all_few = self.bool_train | self.bool_fewshot



        # build index sets for leave one out (loo)
        self.looset = []
        for i in range(self.N):
            loo = {}
            loo["i"] = i
            loo["not_i"] = np.array(
                [
                    ii for ii in np.arange(self.N) if not ii==i
                ]
            )
            index_few_1 = self.indices_fewshot[loo["i"]]
            indices_few_not_1 = self.indices_fewshot[loo["not_i"]]
            
            # for 1-class training set in this loo-validation
            loo["_bool_train_1cls"] = self.bool_train2_for_loo_training

            # for 2-class training set in this loo-validation
            loo["_bool_train_2cls"] = self.bool_train2_for_loo_training | binarize(indices_few_not_1)

            loo["_bool_valid"] = self.bool_train1_for_loo_evaluation | binarize(index_few_1)

            self.looset.append(loo)
    
    def do_validation(self, train_metafeatures, test_metafeatures, descriptor_base):
        # do validation with various zero-shot or few-shot classifiers

        # concatenate metafeatures and labels
        metafeatures = np.concatenate((train_metafeatures, test_metafeatures), axis=0)
        labels = self.labels

        ####
        # zero-shot (one-class) classifiers

        # single feature

        self.selector.add(
            f"{descriptor_base}_recon_loss",
            [ 
                roc_auc_score(
                    labels[loo["_bool_valid"]], 
                    metafeatures[loo["_bool_valid"], 0]
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_genuine_test],
                metafeatures[self.bool_genuine_test, 0]
            )
        )

        # zero-shot (one-class) classifiers
        self.selector.add(
            f"{descriptor_base}_percetual_loss",
            [ 
                roc_auc_score(
                    labels[loo["_bool_valid"]], 
                    metafeatures[loo["_bool_valid"], 1]
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_genuine_test],
                metafeatures[self.bool_genuine_test, 1]
            )
        )

        # zero-shot (one-class) classifiers
        self.selector.add(
            f"{descriptor_base}_range_compression_length",
            [ 
                roc_auc_score(
                    labels[loo["_bool_valid"]], 
                    metafeatures[loo["_bool_valid"], 2]
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_genuine_test],
                metafeatures[self.bool_genuine_test, 2]
            )
        )


        ####
        # for training (zero-shot)

        # Maharanobis

        mcd = MinCovDet()

        mcd.fit(metafeatures[self.bool_train2_for_loo_training, :])

        self.selector.add(
            f"{descriptor_base}_Maharanobis_dist",
            [ 
                roc_auc_score(
                    labels[loo["_bool_valid"]], 
                    mcd.mahalanobis(
                        metafeatures[loo["_bool_valid"], :]
                    )
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_genuine_test],
                mcd.mahalanobis(metafeatures[self.bool_genuine_test, :])
            )
        )

        # ocsvm
        for nu in (0.01, 0.03, 0.1, 0.3):
            ocsvm = sklearn.svm.OneClassSVM(nu=nu)
            pipe = make_pipeline(StandardScaler(), ocsvm)

            pipe.fit(metafeatures[self.bool_train2_for_loo_training, :])

            self.selector.add(
                f'{descriptor_base}_oneclass_svm_nu{nu}', 
                [
                    roc_auc_score(
                        labels[loo["_bool_valid"]], 
                        -pipe.decision_function(
                            metafeatures[loo["_bool_valid"], :]
                        )
                    ) for loo in self.looset
                ],
                roc_auc_score(
                    labels[self.bool_genuine_test], 
                    -pipe.decision_function(metafeatures[self.bool_genuine_test, :])
                )
            )

        # isolation forest
        isof = sklearn.ensemble.IsolationForest()

        isof.fit(metafeatures[self.bool_train2_for_loo_training, :])

        self.selector.add(
            f'{descriptor_base}_isolation_forest', 
            [
                roc_auc_score(
                    labels[loo["_bool_valid"]], 
                    -isof.decision_function(
                        metafeatures[loo["_bool_valid"], :]
                    )
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_genuine_test], 
                -isof.decision_function(metafeatures[self.bool_genuine_test, :])
            )
        )



        ####
        # learnable 2-class validators

        # Mahalanobis ratio

        assert mcd is not None
        
        mcd_anomaly = MinCovDet()
        
        self.selector.add(
            f'{descriptor_base}_Mahalanobis_ratio', 
            [
                roc_auc_score(
                    labels[loo["_bool_valid"]], 
                    mcd.\
                        mahalanobis(metafeatures[loo["_bool_valid"], :]) /\
                    mcd_anomaly.fit(metafeatures[self.indices_few[loo["not_i"]], :]).
                        mahalanobis(metafeatures[loo["_bool_valid"], :])
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_genuine_test], 
                mcd.mahalanobis(metafeatures[self.bool_genuine_test, :]) / \
                mcd_anomaly.fit(metafeatures[self.indices_few, :])     \
                   .mahalanobis(metafeatures[self.bool_genuine_test, :]) 
            )
        )

        # svm
        svc = sklearn.svm.SVC()
        pipe = make_pipeline(StandardScaler(), svc)
        self.selector.add(
            f'{descriptor_base}_SVM',
            [
                roc_auc_score(
                    labels[loo["_bool_valid"]], 
                    pipe.fit(
                        metafeatures[loo["_bool_train_2cls"], :], 
                        labels[loo["_bool_train_2cls"]]
                    ).decision_function(
                        metafeatures[loo["_bool_valid"], :]
                    )
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_genuine_test], 
                pipe.fit(
                    metafeatures[self.bool_all_train_plus_all_few, :],
                    labels[self.bool_all_train_plus_all_few]
                ).decision_function(
                    metafeatures[self.bool_genuine_test]
                )
            )
        )                    

        # weighted svm
        # svm with balancing (training data is also used)
        for C in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e-0, 1.0e+1, 1.0e+2, 1.0e+3, 1.0e+4):
            svc = sklearn.svm.SVC(class_weight='balanced', C=C)
            pipe = make_pipeline(StandardScaler(), svc)
            self.selector.add(
                f'{descriptor_base}_svm_balanced_C={C:.5f}',
                [
                    roc_auc_score(
                        labels[loo["_bool_valid"]], 
                        pipe.fit(
                            metafeatures[loo["_bool_train_2cls"], :], 
                            labels[loo["_bool_train_2cls"]]
                        ).decision_function(
                            metafeatures[loo["_bool_valid"], :]
                        )
                    ) for loo in self.looset
                ],
                roc_auc_score(
                    labels[self.bool_genuine_test], 
                    pipe.fit(
                        metafeatures[self.bool_all_train_plus_all_few, :],
                        labels[self.bool_all_train_plus_all_few]
                    ).decision_function(
                        metafeatures[self.bool_genuine_test]
                    )
                )
            )                    


        # random forest
        rf = sklearn.ensemble.RandomForestClassifier()
        self.selector.add(
            f'{descriptor_base}_RF',
            [
                roc_auc_score(
                    labels[loo["_bool_valid"]], 
                    rf.fit(
                        metafeatures[loo["_bool_train_2cls"], :], 
                        labels[loo["_bool_train_2cls"]]
                    ).predict_proba(
                        metafeatures[loo["_bool_valid"], :]
                    )
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_genuine_test], 
                rf.fit(
                    metafeatures[self.bool_all_train_plus_all_few, :],
                    labels[self.bool_all_train_plus_all_few]
                ).predict_proba(
                    metafeatures[self.bool_genuine_test]
                )
            )
        )                    



        res = self.selector.get_current_best()

        print(res)

        return res["best_avg_rank"], res["peeked_current_real_best_test_score"], res["best_model_descriptor"], res["test_score_with_the_best"]
