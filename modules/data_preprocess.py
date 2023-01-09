from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class LoanDataPreprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        self.float_features_ = [
            'id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
            'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs',
            'inq_last_6mths', 'mths_since_last_delinq',
            'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
            'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv',
            'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
            'total_rec_int', 'total_rec_late_fee', 'recoveries',
            'collection_recovery_fee', 'last_pymnt_amnt',
            'collections_12_mths_ex_med', 'mths_since_last_major_derog',
            'policy_code', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq',
            'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_12m',
            'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
            'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
            'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m'
            ]
        self.int_features_ = [
            'id', 'member_id', 'delinq_2yrs', 'open_acc', 'pub_rec',
            'total_acc', 'collections_12_mths_ex_med', 'policy_code',
            'acc_now_delinq'
            ]
        self.datetime_features_ = [
            'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d',
            'last_credit_pull_d'
            ]
        self.string_features_ = ['emp_title', 'title', 'desc', 'zip_code', 'addr_state']
        self.categorical_features_ = [
            'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
            'verification_status', 'loan_status', 'pymnt_plan', 'purpose',
            'initial_list_status', 'application_type', 'verification_status_joint'
            ]
        self.emp_length_order_ = [
            '< 1 year', '1 year', '2 years', '3 years', '4 years', 
            '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'
            ]
        return self
        
    def transform(self, X, y=None):
        # numerical datatype casting
        for int_feature in self.int_features_ :
            X.loc[:, int_feature] = X.loc[:, int_feature].astype('int64', errors='ignore')
        for float_feature in self.float_features_:
            X.loc[:, float_feature] = X.loc[:, float_feature].astype('float64', errors='ignore')
        # strip string
        for string_feature in self.string_features_:
            X.loc[:, string_feature] = X.loc[:, string_feature].str.strip()
        # datetime datatype casting
        for datetime_feature in self.datetime_features_:
            X.loc[:, datetime_feature] = \
                pd.to_datetime(X.loc[:, datetime_feature], format='%b-%Y')
        # categorical datatype casting
        for categorical_feature in self.categorical_features_:
            X.loc[:, categorical_feature] = \
                X.loc[:, categorical_feature].astype('category', errors='ignore')
        # ordinal category order
        X['emp_length'] = pd.Categorical(
            values=X['emp_length'],
            categories=self.emp_length_order_,
            ordered=True
            )
        return X
