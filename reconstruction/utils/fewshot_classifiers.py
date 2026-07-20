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



__multiplier = 10



class BestModelSelector():
    def __init__(self, n_fewshot : int):
        self.N = n_fewshot
        self.data = []
        self.best_test_score = -np.inf
    
    def add(description : str, fewshot_scores : list[float], test_score : float):
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
    
    def get_current_best():
        # calculate ranks
        rank_matrix = np.ndarray(self.N, len(self.data))
        
        for j, d in enumerate(self.data)):
            rank_matrix[:, j] = d["fewshot_scores"]
        
        for i in range(self.N):
            rank_matrix[i, :] = np.argsort(np.argsort(rank_matrix[i, :]))
        
        # calculate sum rank
        avg_rank = np.sum(rank_matrix, axis=0)
        
        # best model
        best_j = np.argmax(avg_rank)
        
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
        
        #integrate train and test datasets
        self.n_total = len(train_labels) + len(test_labels)
        self.labels = np.concatenate((train_labels, test_labels), axis=0)
        
        random.seed(random_seed)

        # build integrated binary and index set

        # pickup N positives from the test data (for LOO)
        self.indices_positive_in_test = np.where(test_labels!=0)
        self.indices_validatortrainer_in_test = random.sample(indices_positive_in_test, self.N)
        self.indices_tester_in_test = np.array(
            [
                ii for ii in np.arange(len(test_labels)) if not in self.indices_validatortrainer_in_test
            ]
        )

        # pickup N*10 negatives from the train data (for LOO)
        self.indices_negative_in_train = np.where(train_labels==0)
        self.indices_validatortrainer_in_train = random.sample(indices_negative_in_train, self.N * __multiplier)
        self.indices_trainer_in_train = np.array(
            [
                ii for ii in np.arange(len(train_labels)) if not in self.indices_validatortrainer_in_train
            ]
        )

        self.n_train = len(train_labels)
        self.n_test = len(test_labels)

        # for base train (not for LOO)
        self.bool_train_rest = np.array(
            [
                (ii not in self.indices_validatortrainer_in_train) 
                    for ii in np.arange(self.n_train)
            ] + [0,]*self.n_test
        ).astype(np.bool)

        # for base test (not for LOO)
        self.bool_test_rest = np.array(
            [0,]*self.n_train + [
                (ii not in self.indices_validatortrainer_in_test) 
                    for ii in np.arange(self.n_test)
            ]
        ).astype(np.bool)

        # not for LOO verification
        self.bool_rest = self.bool_train_rest | self.bool_test_rest

        # samples removed from original train dataset
        self.indices_few =  np.arange(self.n_total)[self.indices_validatortrainer_in_train]

        # final trainable set (not for LOO)
        self.bool_all_train_plus_all_few = np.copy(self.bool_test_rest)
        self.bool_all_train_plus_all_few[0:self.n_train] = True

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
            index_few_1 = self.indices_few[loo["i"]]
            indices_few_not_1 = self.indices_few[loo["not_i"]]
            
            # for 1-class training set in this loo-validation
            loo["_bool_train_1cls"] = np.copy(self.bool_train_rest)

            # for 2-class training set in this loo-validation

            tmp = np.copy(self.bool_train_rest)
            tmp[indices_few_not_1] = True
            loo["_bool_train_2cls"] = tmp

            #evaluation set in this loo-validation
            tmp = np.copy(self.bool_test_rest)
            tmp[index_few_1] = True
            loo["_bool_test"] = tmp

            self.looset.append(loo)
    
    def do_validation(train_metafeatures, test_metafeatures, descriptor_base):
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
                    labels[loo["_bool_train_1cls"]], 
                    metafeatures[loo["bool_train_1cls"], 0]
                ) for loo in self.looset)
            ],
            roc_auc_score(
                labels[self.bool_test_rest],
                metafeatures[self.bool_test_rest, 0]
        )

        # zero-shot (one-class) classifiers
        self.selector.add(
            f"{descriptor_base}_percetual_loss",
            [ 
                roc_auc_score(
                    labels[loo["_bool_train_1cls"]], 
                    metafeatures[loo["bool_train_1cls"], 1]
                ) for loo in self.looset)
            ],
            roc_auc_score(
                labels[self.bool_test_rest],
                metafeatures[self.bool_test_rest, 1]
        )

        # zero-shot (one-class) classifiers
        self.selector.add(
            f"{descriptor_base}_range_compression_length",
            [ 
                roc_auc_score(
                    labels[loo["_bool_train_1cls"]], 
                    metafeatures[loo["bool_train_1cls"], 2]
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_test_rest],
                metafeatures[self.bool_test_rest, 2]
        )


        ####
        # for training (zero-shot)

        # Maharanobis

        mcd = MinCovDet()

        mcd.fit(metafeatures[loo["bool_train_1cls"], :])

        self.selector.add(
            f"{descriptor_base}_Maharanobis_dist",
            [ 
                roc_auc_score(
                    labels[loo["_bool_train_1cls"]], 
                    mcd.mahalanobis(
                        metafeatures[loo["bool_train_1cls"], :]
                    )
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_test_rest],
                mcd.mahalanobis(metafeatures[self.bool_test_rest, :])
            )
        )

        # ocsvm
        for nu in (0.01, 0.03, 0.1, 0.3):
            ocsvm = sklearn.svm.OneClassSVM(nu=nu)
            pipe = make_pipeline(StandardScaler(), ocsvm)

            pipe.fit(metafeatures[loo["bool_train_1cls"], :])

            self.selector.add(
                f'{descriptor_base}_oneclass_svm_nu{nu}', 
                [
                    roc_auc_score(
                        labels[loo["_bool_train_1cls"]], 
                        -pipe.decision_function(
                            metafeatures[loo["_bool_train_1cls"], :]
                        )
                    ) for loo in self.looset
                ],
                roc_auc_score(
                    labels[self.bool_test_rest], 
                    mcd.mahalanobis(metafeatures[self.bool_test_rest, :])
                )
            )

        # isolation forest
        isof = sklearn.ensemble.IsolationForest()

        isof.fit(metafeatures[loo["bool_train_1cls"], :])

        self.selector.add(
            f'{descriptor_base}_isolation_forest', 
            [
                roc_auc_score(
                    metafeatures[loo["_bool_train_1cls"][loo["not_i"]], :], 
                    -isof.decision_function(
                        metafeatures[loo["_bool_train_1cls"], :]
                    )
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_test_rest], 
                mcd.mahalanobis(metafeatures[self.bool_test_rest, :])
            )
        )



        ####
        # learnable 2-class validators

        # Mahalanobis ratio

        assert mcd is not None
        scores = []
        
        mcd_anomaly = MinCovDet()
        
        self.selector.add(
            f'{descriptor_base}_Mahalanobis_ratio', 
            [
                roc_auc_score(
                    labels[loo["_bool_train_2cls"]], 
                    mcd.
                        mahalanobis(metafeatures[loo["_bool_train_2cls"], :]) /\
                    mcd_anomaly.fit(metafeatures[self.indices_few[loo["not_i"]]], :]).
                        mahalanobis(metafeatures[loo["_bool_train_2cls"], :])
                    )
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_test_rest], 
                mcd.mahalanobis(metafeatures[self.bool_test_rest, :]) / \
                mcd_anomaly.fit(metafeatures[self.indices_few], :])     \
                   .mahalanobis(metafeatures[self.bool_test_rest, :]) 
            )
        )

        # svm
        svc = sklearn.svm.SVC()
        pipe = make_pipeline(StandardScaler(), svc)
        self.selector.add(
            f'{descriptor_base}_SVM',
            [
                roc_auc_score(
                    labels[loo["_bool_test"]], 
                    pipe.fit(
                        metafeatures[loo["_bool_train_2cls"], :], 
                        label[loo["_bool_train_2cls"], :]
                    ).decision_function(
                        metafeatures[loo["_bool_test"], :]
                    )
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_test_rest], 
                pipe.fit(
                    metafeatures[self.bool_all_train_plus_all_few, :],
                    labels[self.bool_all_train_plus_all_few]
                ).decision_function(
                    metafeatures[self.bool_test_rest]
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
                        labels[loo["_bool_test"]], 
                        pipe.fit(
                            metafeatures[loo["_bool_train_2cls"], :], 
                            label[loo["_bool_train_2cls"], :]
                        ).decision_function(
                            metafeatures[loo["_bool_test"], :]
                        )
                    ) for loo in self.looset
                ],
                roc_auc_score(
                    labels[self.bool_test_rest], 
                    pipe.fit(
                        metafeatures[self.bool_all_train_plus_all_few, :],
                        labels[self.bool_all_train_plus_all_few]
                    ).decision_function(
                        metafeatures[self.bool_test_rest]
                    )
                )
            )                    


        # random forest
        rf = sklearn.ensemble.RandomForestClassifier()
        self.selector.add(
            f'{descriptor_base}_RF',
            [
                roc_auc_score(
                    labels[loo["_bool_test"]], 
                    rf.fit(
                        metafeatures[loo["_bool_train_2cls"], :], 
                        label[loo["_bool_train_2cls"], :]
                    ).predict_proba(
                        metafeatures[loo["_bool_test"], :]
                    )
                ) for loo in self.looset
            ],
            roc_auc_score(
                labels[self.bool_test_rest], 
                rf.fit(
                    metafeatures[self.bool_all_train_plus_all_few, :],
                    labels[self.bool_all_train_plus_all_few]
                ).predict_proba(
                    metafeatures[self.bool_test_rest]
                )
            )
        )                    



        res = self.selector.get_current_best()

        print(res)

        return res["best_avg_rank"], res["peeked_current_real_best_test_score"], res["best_model_descriptor"]
