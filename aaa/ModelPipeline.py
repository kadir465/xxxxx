
import numpy as np
import pandas as pd
from numpy.ma.extras import apply_along_axis

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import joblib

cols_to_drop = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_vet_stat', 'koi_vet_date', 'koi_disp_prov', 'koi_comment',
                'koi_period_err1', 'koi_period_err2', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_time0_err1',
                'koi_time0_err2', 'koi_eccen', 'koi_eccen_err1', 'koi_eccen_err2', 'koi_longp', 'koi_longp_err1',
                'koi_longp_err2', 'koi_ingress', 'koi_ingress_err1', 'koi_ingress_err2', 'koi_incl_err1',
                'koi_incl_err2', 'koi_teq_err1', 'koi_teq_err2', 'koi_limbdark_mod', 'koi_parm_prov',
                'koi_tce_plnt_num', 'koi_tce_delivname', 'koi_quarters', 'koi_trans_mod', 'koi_model_dof',
                'koi_model_chisq', 'koi_smet', 'koi_sage', 'koi_sparprov', 'koi_kepmag', 'koi_gmag', 'koi_rmag',
                'koi_imag', 'koi_zmag', 'koi_jmag', 'koi_hmag', 'koi_kmag', 'koi_datalink_dvs', 'koi_datalink_dvr',
                'koi_sage_err1', 'koi_sage_err2', 'koi_sma_err1', 'koi_sma_err2', 'koi_pdisposition']

categorical_features = ['koi_fittype']
numerical_features = ['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
                      'koi_period', 'koi_time0bk', 'koi_time0', 'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
                      'koi_duration', 'koi_duration_err1', 'koi_duration_err2', 'koi_depth', 'koi_depth_err1',
                      'koi_depth_err2', 'koi_ror', 'koi_ror_err1', 'koi_ror_err2', 'koi_srho', 'koi_srho_err1',
                      'koi_srho_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2', 'koi_sma', 'koi_incl', 'koi_teq',
                      'koi_insol', 'koi_insol_err1', 'koi_insol_err2', 'koi_dor', 'koi_dor_err1', 'koi_dor_err2',
                      'koi_ldm_coeff4', 'koi_ldm_coeff3', 'koi_ldm_coeff2', 'koi_ldm_coeff1', 'koi_max_sngle_ev',
                      'koi_max_mult_ev', 'koi_model_snr', 'koi_count', 'koi_num_transits', 'koi_bin_oedp_sig',
                      'koi_steff',
                      'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2',
                      'koi_smet_err1',
                      'koi_smet_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'koi_smass', 'koi_smass_err1',
                      'koi_smass_err2', 'ra', 'dec', 'koi_fwm_stat_sig', 'koi_fwm_sra', 'koi_fwm_sra_err',
                      'koi_fwm_sdec',
                      'koi_fwm_sdec_err', 'koi_fwm_srao', 'koi_fwm_srao_err', 'koi_fwm_sdeco', 'koi_fwm_sdeco_err',
                      'koi_fwm_prao', 'koi_fwm_prao_err', 'koi_fwm_pdeco', 'koi_fwm_pdeco_err', 'koi_dicco_mra',
                      'koi_dicco_mra_err', 'koi_dicco_mdec', 'koi_dicco_mdec_err', 'koi_dicco_msky',
                      'koi_dicco_msky_err',
                      'koi_dikco_mra', 'koi_dikco_mra_err', 'koi_dikco_mdec', 'koi_dikco_mdec_err', 'koi_dikco_msky',
                      'koi_dikco_msky_err']


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X ,y):
        return self

    def transform(self, X):
        return X.drop(self.cols_to_drop, axis=1, errors='ignore')


def create_pipeline(model):
    numerical_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_preprocessor, numerical_features),
        ('cat', categorical_preprocessor, categorical_features)
    ])

    pipeline = Pipeline([
        ('dropper', DropColumns(cols_to_drop)),
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline
