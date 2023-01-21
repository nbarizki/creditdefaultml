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

class LoanDataLabelPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, include=[], exclude=[]):
        self.include = include
        self.exclude = exclude

    def fit(self, X=None, y=None):
        self.applicant_features_ = [
            'emp_length', 'home_ownership',
            'annual_inc', 'verification_status', 'dti', 'delinq_2yrs',
            'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq',
            'mths_since_last_record', 'open_acc', 'revol_bal', 'revol_util',
            'total_acc', 'collections_12_mths_ex_med', 'mths_since_last_major_derog',
            'policy_code', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 
            'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
            'il_util', 'open_rv_12m', 'open_rv_24m', 'all_util', 'total_rev_hi_lim', 
            'inq_fi', 'total_cu_tl', 'inq_last_12m', 'pub_rec', 'max_bal_bc'
            ]
        self.loan_features_ = [
            'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
            'sub_grade', 'pymnt_plan',
            ]
        self.post_origin_features_ = [
            'funded_amnt', 'funded_amnt_inv', 'issue_d', 'loan_status', 
            'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_rec_prncp',
            'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
            'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
             'total_pymnt', 'total_pymnt_inv'
            ]
        self.mapping_loan_cat_ = {
            'Charged Off': 'Bad Loan', 
            'Late (31-120 days)': 'Bad Loan', 
            'In Grace Period': 'Bad Loan', 
            'Late (16-30 days)': 'Bad Loan', 
            'Default': 'Bad Loan', 
            'Does not meet the credit policy. Status:Charged Off': 'Bad Loan',
            'Fully Paid': 'Good Loan', 
            'Does not meet the credit policy. Status:Fully Paid': 'Good Loan'
            }
        return self
    
    def transform(self, X, y=None):
        # Filter to only have 'INDIVIDUAL'
        X = X[X.application_type.str.upper() == 'INDIVIDUAL']
        # Create Label
        X = X.assign(
            loan_category=X.loan_status.map(self.mapping_loan_cat_)
            )
        # Filter to only have Good Loan and Bad Loan
        X = X[X.loan_category.isin(['Good Loan', 'Bad Loan'])].reset_index()
        # return only selected 
        if not self.exclude:
            return X[['loan_category']
                     + self.applicant_features_ 
                     + self.loan_features_ + self.include],
        else:
            return X[['loan_category']
                     + self.applicant_features_ 
                     + self.loan_features_ + self.include].drop(columns=self.exclude)

class LoanDataMissingHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def _perform(self, masking, feature=None, replace_value=None, drop_obs=False):
        if replace_value:
            self.X.loc[masking, feature] = replace_value
        if drop_obs:
            self.X = self.X.drop(self.X[masking].index)

    def fit(self, X, y=None):
        self.median_cond_3_ = X.loc[
            (X.delinq_2yrs > 0) & (X.acc_now_delinq == 0),
            'mths_since_last_delinq'].median()
        return self 

    def transform(self, X, y=None):
        self.X = X
        # mths_since_last_record
        rec_condition_1 = (self.X.mths_since_last_record.isna()) & (self.X.pub_rec == 0)
        self._perform(rec_condition_1, 'mths_since_last_record', 99)
        rec_condition_2 = (self.X.mths_since_last_record.isna()) & (self.X.pub_rec > 0)
        self._perform(rec_condition_2, 'mths_since_last_record', 1)
        rec_condition_3 = (self.X.mths_since_last_record.isna()) & (self.X.pub_rec.isna())
        self._perform(rec_condition_3, drop_obs=True)
        # mths_since_last_delinq
        last_delinq_condition_1 = \
            (self.X.mths_since_last_delinq.isna())\
            & (self.X.delinq_2yrs == 0)\
            & (self.X.acc_now_delinq == 0)
        self._perform(last_delinq_condition_1, 'mths_since_last_delinq', 99)
        last_delinq_condition_2 = \
            (self.X.mths_since_last_delinq.isna())\
            & (self.X.delinq_2yrs > 0)\
            & (self.X.acc_now_delinq > 0)
        self._perform(last_delinq_condition_2, 'mths_since_last_delinq', 1)
        last_delinq_condition_3 = \
            (self.X.mths_since_last_delinq.isna())\
            & (self.X.delinq_2yrs > 0)\
            & (self.X.acc_now_delinq == 0)
        self._perform(last_delinq_condition_3, 'mths_since_last_delinq', self.median_cond_3_)
        last_delinq_condition_4 = \
            (self.X.mths_since_last_delinq == 0)\
            & (self.X.delinq_2yrs > 0)\
            & (self.X.acc_now_delinq == 0)
        self._perform(last_delinq_condition_4, 'mths_since_last_delinq', self.median_cond_3_)
        last_delinq_condition_5 = \
            (self.X.mths_since_last_delinq.between(0, 25, inclusive='neither'))\
            & (self.X.delinq_2yrs == 0)\
            & (self.X.acc_now_delinq == 0)
        self._perform(last_delinq_condition_5, 'delinq_2yrs', 1)
        last_delinq_condition_6 = \
            (self.X.mths_since_last_delinq == 0)\
            & (self.X.delinq_2yrs == 0)\
            & (self.X.acc_now_delinq == 0)
        self._perform(last_delinq_condition_6, 'mths_since_last_delinq', 99)
        last_delinq_condition_7 = \
            (self.X.mths_since_last_delinq.isna())\
            & (self.X.delinq_2yrs.isna())\
            & (self.X.acc_now_delinq.isna())
        self._perform(last_delinq_condition_7, drop_obs=True)
        # inq_last_{6mths, 12mths}
        inq_condition_1 = \
            (self.X.inq_last_6mths.isna() | self.X.inq_last_12m.isna())\
            & (self.X.inq_fi == 0)
        self._perform(inq_condition_1, ['inq_last_6mths', 'inq_last_12m'], 0)
        inq_condition_2 = \
            (self.X.inq_last_6mths.isna() | self.X.inq_last_12m.isna())\
            & (self.X.inq_fi > 0)
        try:
            self._perform(inq_condition_2, 'inq_last_6mths', self.X.loc[inq_condition_2, 'inq_fi'])
            self._perform(inq_condition_2, 'inq_last_12m', self.X.loc[inq_condition_2, 'inq_fi'])
        except ValueError:
            pass
        inq_condition_3 = \
            (self.X.inq_last_6mths.isna() & self.X.inq_last_12m.isna())\
            & (self.X.inq_fi.isna())
        self._perform(inq_condition_3, drop_obs=True)
        # open_acc
        acc_condition_1_2 = \
            (self.X.open_acc.isna() | self.X.total_acc.isna())\
            & ((self.X.open_acc == 0) | (self.X.total_acc == 0))
        self._perform(acc_condition_1_2, 'open_acc', 0)
        self._perform(acc_condition_1_2, 'total_acc', 0)
        acc_condition_3_4 = \
            (self.X.open_acc.isna() | self.X.total_acc.isna())\
            & ((self.X.open_acc > 0) | (self.X.total_acc > 0))
        self._perform(acc_condition_3_4, drop_obs=True)
        acc_condition_5 = \
            self.X.open_acc.isna() & self.X.total_acc.isna()
        self._perform(acc_condition_5, drop_obs=True)
        # total_acc, tot_cur_bar
        bal_condition_1_2 = \
            (self.X.tot_cur_bal.isna() | self.X.total_acc.isna())\
            & ((self.X.tot_cur_bal == 0) | (self.X.total_acc == 0))
        self._perform(bal_condition_1_2, 'total_acc', 0)
        self._perform(bal_condition_1_2, 'tot_cur_bal', 0)
        bal_condition_3_4 = \
            (self.X.tot_cur_bal.isna() | self.X.total_acc.isna())\
            & ((self.X.tot_cur_bal > 0) | (self.X.total_acc > 0))
        self._perform(bal_condition_3_4, drop_obs=True)
        bal_condition_5 = \
            self.X.tot_cur_bal.isna() & self.X.total_acc.isna()
        self._perform(bal_condition_5, drop_obs=True)
        # total_coll_amnt
        coll_condition_1 = \
            self.X.tot_coll_amt.isna() & self.X.collections_12_mths_ex_med.isna()
        self._perform(coll_condition_1, drop_obs=True)
        coll_condition_2_3 = \
            (self.X.tot_coll_amt.isna() | (self.X.tot_coll_amt == 0))\
            & (self.X.collections_12_mths_ex_med.isna() | (self.X.collections_12_mths_ex_med == 0))
        self._perform(coll_condition_2_3, 'tot_coll_amt', 0)
        self._perform(coll_condition_2_3, 'collections_12_mths_ex_med', 0)
        coll_condition_4 = \
            self.X.tot_coll_amt.isna() & (self.X.collections_12_mths_ex_med > 0)
        self._perform(coll_condition_4, drop_obs=True)
        coll_condition_5 = \
            (self.X.tot_coll_amt == 0) & (self.X.collections_12_mths_ex_med.isna())
        self._perform(coll_condition_5, drop_obs=True)
        # open_il
        open_il_condition_1 = \
            (self.X.open_il_12m.isna() | self.X.open_il_24m.isna())\
            & (self.X.mths_since_rcnt_il.isna())\
            & (self.X.total_bal_il == 0)
        self._perform(open_il_condition_1, 'mths_since_rcnt_il', 99)
        self._perform(open_il_condition_1, 'open_il_12m', 0)
        self._perform(open_il_condition_1, 'open_il_24m', 0)
        open_il_condition_2 = \
            (self.X.open_il_12m.isna() | self.X.open_il_24m.isna())\
            & (self.X.mths_since_rcnt_il.isna())\
            & (self.X.total_bal_il > 0)
        self._perform(open_il_condition_2, drop_obs=True)
        open_il_condition_3 = \
            ((self.X.open_il_12m > 0) | (self.X.open_il_24m > 0))\
            & (self.X.mths_since_rcnt_il.isna())\
            & (self.X.total_bal_il > 0)
        self._perform(open_il_condition_3, 'mths_since_rcnt_il', 1)
        open_il_condition_4 = \
            (self.X.open_il_12m.isna() & self.X.open_il_24m.isna())\
            & (self.X.mths_since_rcnt_il.isna())\
            & (self.X.total_bal_il.isna())
        self._perform(open_il_condition_4, 'open_il_12m', 0) 
        self._perform(open_il_condition_4, 'open_il_24m', 0) 
        self._perform(open_il_condition_4, 'mths_since_rcnt_il', 99) 
        self._perform(open_il_condition_4, 'total_bal_il', 0)
        # total_bal_il
        bal_il_condition_1 = \
            (self.X.total_bal_il > 0)\
            & (self.X.il_util.isna())
        self._perform(bal_il_condition_1, drop_obs=True)
        bal_il_condition_2 = \
            (self.X.total_bal_il.isna())\
            & (self.X.il_util > 0)
        self._perform(bal_il_condition_2, drop_obs=True)
        bal_il_condition_3 = \
            (self.X.total_bal_il.isna())\
            & (self.X.il_util == 0)
        self._perform(bal_il_condition_3, 'total_bal_il', 0)
        bal_il_condition_4 = \
            (self.X.total_bal_il == 0)\
            & (self.X.il_util.isna())
        self._perform(bal_il_condition_4, 'il_util', 0)
        bal_il_condition_5 = \
            (self.X.total_bal_il.isna())\
            & (self.X.il_util.isna())
        self._perform(bal_il_condition_5, 'total_bal_il', 0)
        self._perform(bal_il_condition_5, 'il_util', 0)
        # revol_bal
        revol_condition_1 = \
            (self.X.revol_bal > 0)\
            & (self.X.revol_util.isna())
        self._perform(revol_condition_1, drop_obs=True)
        revol_condition_2 = \
            (self.X.revol_bal.isna())\
            & (self.X.revol_util > 0)
        self._perform(revol_condition_2, drop_obs=True)
        revol_condition_3 = \
            (self.X.revol_bal.isna())\
            & (self.X.revol_util == 0)
        self._perform(revol_condition_3, 'revol_bal', 0)
        revol_condition_4 = \
            (self.X.revol_bal == 0)\
            & (self.X.revol_util.isna())
        self._perform(revol_condition_4, 'revol_util', 0)
        revol_condition_5 = \
            (self.X.revol_bal.isna())\
            & (self.X.revol_util.isna())
        self._perform(revol_condition_5, 'revol_bal', 0)
        self._perform(revol_condition_5, 'revol_util', 0)
        ## Mandatory Features
        mandatory_features = [
            'annual_inc', 'dti', 'home_ownership',
            'loan_amnt', 'term', 'int_rate', 'installment', 
            'grade', 'sub_grade', 'pymnt_plan'
            ]
        for feature in mandatory_features:
            try:
                condition = self.X[feature].isna()
                self._perform(condition, drop_obs=True)
            except KeyError:
                continue
        ## replace missing values outside special conditions
        # replace 0
        replace_0 = [
            'il_util', 'total_cu_tl', 'inq_last_12m', 'all_util', 'open_rv_24m', 'open_rv_12m', 'open_acc_6m', 
            'total_bal_il', 'inq_fi', 'open_il_12m', 'open_il_24m', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
            'revol_util', 'collections_12_mths_ex_med', 'pub_rec', 'acc_now_delinq', 'total_acc', 'open_acc', 'inq_last_6mths',
            'delinq_2yrs', 'max_bal_bc'
            ]
        for feature in replace_0:
            try:
                self.X.loc[:, feature] = self.X.loc[:, feature].fillna(0)
            except KeyError:
                continue
        replace_99 = [
            'mths_since_rcnt_il', 'mths_since_last_record', 'mths_since_last_major_derog', 'mths_since_last_delinq',
            ]
        for feature in replace_99:
            try:
                self.X.loc[:, feature] = self.X.loc[:, feature].fillna(99)
            except KeyError:
                continue
        label = self.X.loan_category.values
        predictor = self.X.drop(columns=['loan_category']).reset_index(drop=True)
        return predictor, label

# class LoanDataMissingHandler(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass

#     def _perform(self, masking, feature=None, replace_value=None, drop_obs=False):
#         nonlocal X
#         if replace_value:
#             X.loc[masking, feature] = replace_value
#         if drop_obs:
#             X = X.drop(X[masking].index)

#     def fit(self, X, y=None):
#         self.median_cond_3_ = X.loc[
#             (X.delinq_2yrs > 0) & (X.acc_now_delinq == 0),
#             'mths_since_last_delinq'].median()
#         return self 

#     def transform(self, X, y=None):
#         self.X = 
#         # mths_since_last_record
#         rec_condition_1 = (X.mths_since_last_record.isna()) & (X.pub_rec == 0)
#         self._perform(rec_condition_1, 'mths_since_last_record', 99)
#         rec_condition_2 = (X.mths_since_last_record.isna()) & (X.pub_rec > 0)
#         self._perform(rec_condition_2, 'mths_since_last_record', 1)
#         rec_condition_3 = (X.mths_since_last_record.isna()) & (X.pub_rec.isna())
#         self._perform(rec_condition_3, drop_obs=True)
#         # mths_since_last_delinq
#         last_delinq_condition_1 = \
#             (X.mths_since_last_delinq.isna())\
#             & (X.delinq_2yrs == 0)\
#             & (X.acc_now_delinq == 0)
#         self._perform(last_delinq_condition_1, 'mths_since_last_delinq', 99)
#         last_delinq_condition_2 = \
#             (X.mths_since_last_delinq.isna())\
#             & (X.delinq_2yrs > 0)\
#             & (X.acc_now_delinq > 0)
#         self._perform(last_delinq_condition_2, 'mths_since_last_delinq', 1)
#         last_delinq_condition_3 = \
#             (X.mths_since_last_delinq.isna())\
#             & (X.delinq_2yrs > 0)\
#             & (X.acc_now_delinq == 0)
#         self._perform(last_delinq_condition_3, 'mths_since_last_delinq', self.median_cond_3_)
#         last_delinq_condition_4 = \
#             (X.mths_since_last_delinq == 0)\
#             & (X.delinq_2yrs > 0)\
#             & (X.acc_now_delinq == 0)
#         self._perform(last_delinq_condition_4, 'mths_since_last_delinq', self.median_cond_3_)
#         last_delinq_condition_5 = \
#             (X.mths_since_last_delinq.between(0, 25, inclusive='neither'))\
#             & (X.delinq_2yrs == 0)\
#             & (X.acc_now_delinq == 0)
#         self._perform(last_delinq_condition_5, 'delinq_2yrs', 1)
#         last_delinq_condition_6 = \
#             (X.mths_since_last_delinq == 0)\
#             & (X.delinq_2yrs == 0)\
#             & (X.acc_now_delinq == 0)
#         self._perform(last_delinq_condition_6, 'mths_since_last_delinq', 99)
#         last_delinq_condition_7 = \
#             (X.mths_since_last_delinq.isna())\
#             & (X.delinq_2yrs.isna())\
#             & (X.acc_now_delinq.isna())
#         self._perform(last_delinq_condition_7, drop_obs=True)
#         # inq_last_{6mths, 12mths}
#         inq_condition_1 = \
#             (X.inq_last_6mths.isna() | X.inq_last_12m.isna())\
#             & (X.inq_fi == 0)
#         self._perform(inq_condition_1, ['inq_last_6mths', 'inq_last_12m'], 0)
#         inq_condition_2 = \
#             (X.inq_last_6mths.isna() | X.inq_last_12m.isna())\
#             & (X.inq_fi > 0)
#         try:
#             self._perform(inq_condition_2, 'inq_last_6mths', X.loc[inq_condition_2, 'inq_fi'])
#             self._perform(inq_condition_2, 'inq_last_12m', X.loc[inq_condition_2, 'inq_fi'])
#         except ValueError:
#             pass
#         inq_condition_3 = \
#             (X.inq_last_6mths.isna() & X.inq_last_12m.isna())\
#             & (X.inq_fi.isna())
#         self._perform(inq_condition_3, drop_obs=True)
#         # open_acc
#         acc_condition_1_2 = \
#             (X.open_acc.isna() | X.total_acc.isna())\
#             & ((X.open_acc == 0) | (X.total_acc == 0))
#         self._perform(acc_condition_1_2, 'open_acc', 0)
#         self._perform(acc_condition_1_2, 'total_acc', 0)
#         acc_condition_3_4 = \
#             (X.open_acc.isna() | X.total_acc.isna())\
#             & ((X.open_acc > 0) | (X.total_acc > 0))
#         self._perform(acc_condition_3_4, drop_obs=True)
#         acc_condition_5 = \
#             X.open_acc.isna() & X.total_acc.isna()
#         self._perform(acc_condition_5, drop_obs=True)
#         # total_acc, tot_cur_bar
#         bal_condition_1_2 = \
#             (X.tot_cur_bal.isna() | X.total_acc.isna())\
#             & ((X.tot_cur_bal == 0) | (X.total_acc == 0))
#         self._perform(bal_condition_1_2, 'total_acc', 0)
#         self._perform(bal_condition_1_2, 'tot_cur_bal', 0)
#         bal_condition_3_4 = \
#             (X.tot_cur_bal.isna() | X.total_acc.isna())\
#             & ((X.tot_cur_bal > 0) | (X.total_acc > 0))
#         self._perform(bal_condition_3_4, drop_obs=True)
#         bal_condition_5 = \
#             X.tot_cur_bal.isna() & X.total_acc.isna()
#         self._perform(bal_condition_5, drop_obs=True)
#         # total_coll_amnt
#         coll_condition_1 = \
#             X.tot_coll_amt.isna() & X.collections_12_mths_ex_med.isna()
#         self._perform(coll_condition_1, drop_obs=True)
#         coll_condition_2_3 = \
#             (X.tot_coll_amt.isna() | (X.tot_coll_amt == 0))\
#             & (X.collections_12_mths_ex_med.isna() | (X.collections_12_mths_ex_med == 0))
#         self._perform(coll_condition_2_3, 'tot_coll_amt', 0)
#         self._perform(coll_condition_2_3, 'collections_12_mths_ex_med', 0)
#         coll_condition_4 = \
#             X.tot_coll_amt.isna() & (X.collections_12_mths_ex_med > 0)
#         self._perform(coll_condition_4, drop_obs=True)
#         coll_condition_5 = \
#             (X.tot_coll_amt == 0) & (X.collections_12_mths_ex_med.isna())
#         self._perform(coll_condition_5, drop_obs=True)
#         # open_il
#         open_il_condition_1 = \
#             (X.open_il_12m.isna() | X.open_il_24m.isna())\
#             & (X.mths_since_rcnt_il.isna())\
#             & (X.total_bal_il == 0)
#         self._perform(open_il_condition_1, 'mths_since_rcnt_il', 99)
#         self._perform(open_il_condition_1, 'open_il_12m', 0)
#         self._perform(open_il_condition_1, 'open_il_24m', 0)
#         open_il_condition_2 = \
#             (X.open_il_12m.isna() | X.open_il_24m.isna())\
#             & (X.mths_since_rcnt_il.isna())\
#             & (X.total_bal_il > 0)
#         self._perform(open_il_condition_2, drop_obs=True)
#         open_il_condition_3 = \
#             ((X.open_il_12m > 0) | (X.open_il_24m > 0))\
#             & (X.mths_since_rcnt_il.isna())\
#             & (X.total_bal_il > 0)
#         self._perform(open_il_condition_3, 'mths_since_rcnt_il', 1)
#         open_il_condition_4 = \
#             (X.open_il_12m.isna() & X.open_il_24m.isna())\
#             & (X.mths_since_rcnt_il.isna())\
#             & (X.total_bal_il.isna())
#         self._perform(open_il_condition_4, 'open_il_12m', 0) 
#         self._perform(open_il_condition_4, 'open_il_24m', 0) 
#         self._perform(open_il_condition_4, 'mths_since_rcnt_il', 99) 
#         self._perform(open_il_condition_4, 'total_bal_il', 0)
#         # total_bal_il
#         bal_il_condition_1 = \
#             (X.total_bal_il > 0)\
#             & (X.il_util.isna())
#         self._perform(bal_il_condition_1, drop_obs=True)
#         bal_il_condition_2 = \
#             (X.total_bal_il.isna())\
#             & (X.il_util > 0)
#         self._perform(bal_il_condition_2, drop_obs=True)
#         bal_il_condition_3 = \
#             (X.total_bal_il.isna())\
#             & (X.il_util == 0)
#         self._perform(bal_il_condition_3, 'total_bal_il', 0)
#         bal_il_condition_4 = \
#             (X.total_bal_il == 0)\
#             & (X.il_util.isna())
#         self._perform(bal_il_condition_4, 'il_util', 0)
#         bal_il_condition_5 = \
#             (X.total_bal_il.isna())\
#             & (X.il_util.isna())
#         self._perform(bal_il_condition_5, 'total_bal_il', 0)
#         self._perform(bal_il_condition_5, 'il_util', 0)
#         # revol_bal
#         revol_condition_1 = \
#             (X.revol_bal > 0)\
#             & (X.revol_util.isna())
#         self._perform(revol_condition_1, drop_obs=True)
#         revol_condition_2 = \
#             (X.revol_bal.isna())\
#             & (X.revol_util > 0)
#         self._perform(revol_condition_2, drop_obs=True)
#         revol_condition_3 = \
#             (X.revol_bal.isna())\
#             & (X.revol_util == 0)
#         self._perform(revol_condition_3, 'revol_bal', 0)
#         revol_condition_4 = \
#             (X.revol_bal == 0)\
#             & (X.revol_util.isna())
#         self._perform(revol_condition_4, 'revol_util', 0)
#         revol_condition_5 = \
#             (X.revol_bal.isna())\
#             & (X.revol_util.isna())
#         self._perform(revol_condition_5, 'revol_bal', 0)
#         self._perform(revol_condition_5, 'revol_util', 0)
#         ## Mandatory Features
#         mandatory_features = [
#             'annual_inc', 'dti', 'home_ownership',
#             'loan_amnt', 'term', 'int_rate', 'installment', 
#             'grade', 'sub_grade', 'pymnt_plan'
#             ]
#         for feature in mandatory_features:
#             condition = X[feature].isna()
#             self._perform(condition, drop_obs=True)
#         ## replace missing values outside special conditions
#         # replace 0
#         replace_0 = [
#             'il_util', 'total_cu_tl', 'inq_last_12m', 'all_util', 'open_rv_24m', 'open_rv_12m', 'open_acc_6m', 
#             'total_bal_il', 'inq_fi', 'open_il_12m', 'open_il_24m', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
#             'revol_util', 'collections_12_mths_ex_med', 'pub_rec', 'acc_now_delinq', 'total_acc', 'open_acc', 'inq_last_6mths',
#             'delinq_2yrs', 'max_bal_bc'
#             ]
#         for feature in replace_0:
#             X.loc[:, feature] = X.loc[:, feature].fillna(0)
#         # replace 99
#         replace_99 = [
#             'mths_since_rcnt_il', 'mths_since_last_record', 'mths_since_last_major_derog', 'mths_since_last_delinq',
#             ]
#         for feature in replace_99:
#             X.loc[:, feature] = X.loc[:, feature].fillna(99)
#         label = X.loan_category.values
#         predictor = X.drop(columns=['loan_category']).reset_index(drop=True)
#         return predictor, label