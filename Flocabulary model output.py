import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from numpy import column_stack
from sklearn.model_selection import train_test_split

source_data ='leadscore_weekly_conversion_data_20180119.csv'

dataframe = pd.read_csv(source_data)

dataframe = dataframe[dataframe['charter_text'] != '']
##sample = dataframe.head(n=1000)

##sample.to_csv ('sample.csv')
target = ['is_paid']

exclude_columns = ['subscription_id',
                   'start_date',
                   'w1',
                   'w2',
                   'w3',
                   'w4',
                   'w5',
                   'w6',
                   'is_paid',
                   'is_current',
                   'trial_duration_granted',
                   'trial_end_date',
                   'end_date',
                   'ismempup',
                   'ispelm',
                   'ispfemale',
                   'ispwhite',
                   'name',
                   'survyear',
                   'fipst',
                   'stabr',
                   'statename',
                   'seaname',
                   'leaid',
                   'st_leaid',
                   'lea_name',
                   'schid',
                   'st_schid',
                   'ncessch',
                   'sch_name',
                   'mstreet1',
                   'mstreet2',
                   'mstreet3',
                   'mcity',
                   'mstate',
                   'mzip',
                   'mzip4',
                   'phone',
                   'lstreet1',
                   'lstreet2',
                   'lstreet3',
                   'lcity',
                   'lstate',
                   'lzip',
                   'lzip4',
                   'union',
                   'out_of_state_flag',
                   'sch_type_text',
                   'sch_type',
                   'recon_status',
                   'gslo',
                   'gshi',
                   'level',
                   'virtual',
                   'bies',
                   'sy_status_text',
                   'sy_status',
                   'updated_status_text',
                   'updated_status',
                   'effective_date',
                   'charter_text',
                   'chartauth1',
                   'chartauthn1',
                   'chartauth2',
                   'chartauthn2',
                   'igoffered',
                   'fte',
                   'totfrl',
                   'frelch',
                   'redlch',
                   'isfle',
                   'titlei',
                   'titlei_text',
                   'titlei_status',
                   'stitlei',
                   'shared_time',
                   'magnet_text',
                   'nslpstatus_text',
                   'nslpstatus_code',
                   'latcode',
                   'longcode',
                   'cd',
                   'locale',
                   'csa',
                   'cbsa',
                   'necta',
                   'metmic'                   ]
categorical_features= ['fl_v33',
'fl_membersch',
'fl_c14',
'fl_c15',
'fl_c16',
'fl_c17',
'fl_c19',
'fl_b11',
'fl_c20',
'fl_c25',
'fl_c36',
'fl_b10',
'fl_b12',
'fl_b13',
'fl_c01',
'fl_c04',
'fl_c05',
'fl_c06',
'fl_c07',
'fl_c08',
'fl_c09',
'fl_c10',
'fl_c11',
'fl_c12',
'fl_c13',
'fl_c35',
'fl_c38',
'fl_c39',
'fl_t02',
'fl_t06',
'fl_t09',
'fl_t15',
'fl_t40',
'fl_t99',
'fl_d11',
'fl_d23',
'fl_a07',
'fl_a08',
'fl_a09',
'fl_a11',
'fl_a13',
'fl_a15',
'fl_a20',
'fl_a40',
'fl_u11',
'fl_u22',
'fl_u30',
'fl_u50',
'fl_u97',
'fl_c24',
'fl_e13',
'fl_v91',
'fl_v92',
'fl_e17',
'fl_e07',
'fl_e08',
'fl_e09',
'fl_v40',
'fl_v45',
'fl_v90',
'fl_v85',
'fl_e11',
'fl_v60',
'fl_v65',
'fl_v70',
'fl_v75',
'fl_v80',
'fl_f12',
'fl_g15',
'fl_k09',
'fl_k10',
'fl_k11',
'fl_l12',
'fl_m12',
'fl_q11',
'fl_i86',
'fl_z32',
'fl_z33',
'fl_z35',
'fl_z36',
'fl_z37',
'fl_z38',
'fl_v11',
'fl_v13',
'fl_v15',
'fl_v17',
'fl_v21',
'fl_v23',
'fl_v37',
'fl_v29',
'fl_z34',
'fl_v10',
'fl_v12',
'fl_v14',
'fl_v16',
'fl_v18',
'fl_v22',
'fl_v24',
'fl_v38',
'fl_v30',
'fl_v32',
'fl_v93',
'fl_19h',
'fl_21f',
'fl_31f',
'fl_41f',
'fl_61v',
'fl_66v',
'fl_w01',
'fl_w31',
'fl_w61',
'fl_hr1',
'fl_he1',
'fl_he2',
                       'pkoffered',
                       'kgoffered',
                       'g1offered',
                       'g2offered',
                       'g3offered',
                       'g4offered',
                       'g5offered',
                       'g6offered',
                       'g7offered',
                       'g8offered',
                       'g9offered',
                       'g10offered',
                       'g11offered',
                       'g12offered',
                       'g13offered',
                       'aeoffered',
                       'ugoffered',
                       'nogrades'                       ]




train_columns = []

#from sklearn.preprocessing import OneHotEncoder
##encoder = OneHotEncoder(categorical_features=categorical_features)
#encoder.fit(dataframe)
##encoded_train = encoder.transform(dataframe)

dataframe_transformed = pd.get_dummies(dataframe, columns=categorical_features)



all_columns=  list(dataframe_transformed )
for column in all_columns:
    if column not in exclude_columns and column not in categorical_features:
        train_columns.append(column)


##dataframe_transformed[train_columns + target].to_pickle ("dataframe_transformed.pkl")
##dataframe[target].to_pickle ("dataframe_target.pkl")

##dataframe_transformed.reset_index()

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn import linear_model, decomposition, datasets
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest



dataframe_transformed_train = dataframe_transformed[train_columns].apply(lambda x: x.fillna(x.mean()),axis=0)

selector = SelectKBest(f_classif, k=50)  # scoring_func like "f1_macro"
selector.fit(dataframe_transformed_train, dataframe_transformed['is_paid'])


names = dataframe_transformed_train.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending= [False,True])

print "best features are", (ns_df_sorted)

#from sklearn.preprocessing import Imputer

#dataframe_transformed = Imputer().fit_transform(dataframe_transformed_train)

features_standardized = MinMaxScaler().fit_transform(dataframe_transformed_train)

##dataframe_transformed.reset_index()




X_train, X_test, y_train, y_test = train_test_split(features_standardized, dataframe_transformed['is_paid'], test_size=0.4, random_state=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn import linear_model, decomposition, datasets








#New dataframe with the selected features only for later use in the classifier.



pca = decomposition.PCA()

parameters_to_tune = {#'pca__n_components':n_components,
                     'kbest__k':[50],
                     #'pca__n_components': [3,5,10]

}

pipe = Pipeline(steps=[('kbest', SelectKBest(f_classif)),('forest',RandomForestClassifier())])
#pipe = Pipeline(steps=[('kbest', SelectKBest(f_classif)),('pca', pca),('forest',RandomForestClassifier())])
#pipe = Pipeline(steps=[('pca', pca.fit_transform()),('forest',RandomForestClassifier())])
estimator=GridSearchCV(pipe,parameters_to_tune, refit=True,scoring='recall_macro')





clf = estimator.fit(X_train, y_train)

pred_tree = estimator.predict(X_test)

print "best estimators are", clf.best_estimator_

from sklearn.metrics import classification_report

print(classification_report(y_test, pred_tree))


names = dataframe_transformed_train.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending= [False,True])

print "best features are",ns_df_sorted

#from sklearn.preprocessing import Imputer

#dataframe_transformed = Imputer().fit_transform(dataframe_transformed_train)

features_standardized = MinMaxScaler().fit_transform(dataframe_transformed_train)

##dataframe_transformed.reset_index()




X_train, X_test, y_train, y_test = train_test_split(features_standardized, dataframe_transformed['is_paid'], test_size=0.4, random_state=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn import linear_model, decomposition, datasets








#New dataframe with the selected features only for later use in the classifier.



pca = decomposition.PCA()

parameters_to_tune = {#'pca__n_components':n_components,
                     'kbest__k':[50],
                     #'pca__n_components': [3,5,10]

}

pipe = Pipeline(steps=[('kbest', SelectKBest(f_classif)),('forest',RandomForestClassifier())])
#pipe = Pipeline(steps=[('kbest', SelectKBest(f_classif)),('pca', pca),('forest',RandomForestClassifier())])
#pipe = Pipeline(steps=[('pca', pca.fit_transform()),('forest',RandomForestClassifier())])
estimator=GridSearchCV(pipe,parameters_to_tune, refit=True,scoring='recall_macro')





clf = estimator.fit(X_train, y_train)

pred_tree = estimator.predict(X_test)

print "best estimators are", clf.best_estimator_

from sklearn.metrics import classification_report

print(classification_report(y_test, pred_tree))


#pred_tree_all = estimator.predict(features_standardized)

x = estimator.predict_proba(features_standardized)


dataframe['prediction0'] = x[:,0]
dataframe['prediction1'] = x[:,1]

output_features = ['subscription_id','prediction0','prediction1']
dataframe_nonpaid = dataframe[dataframe['is_paid']==0]
dataframe[output_features].sort(columns='predictions1', axis=0, ascending=False)
output_prediction_probability= dataframe[output_features]

output_prediction_probability.to_csv("output_predictions.csv")

