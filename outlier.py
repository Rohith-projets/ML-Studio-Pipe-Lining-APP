import streamlit as st
import pandas as pd
from feature_engine.outliers import OutlierTrimmer, Winsorizer

class OutlierTrim:
    def __init__(self, dataset):
        self.dataset = dataset

    def common_parameters(self):
        # Streamlit widgets for user inputs
        cols = st.multiselect('Select Columns to Process', options=self.dataset.select_dtypes(include=['float64', 'int64']).columns.tolist())  # Select columns
        return cols

    def outlier_trimmer(self):
        cols = self.common_parameters()
        capping_method = st.selectbox('Select Outlier Detection Method', ['gaussian', 'iqr', 'mad', 'quantiles'], index=0)
        tail = st.selectbox('Select Tail for Outliers', ['left', 'right', 'both'], index=1)
        fold = st.slider('Select Fold Value', 1.0, 5.0, 3.0, 0.1)
        missing_values = st.selectbox('Missing Values Handling', ['raise', 'ignore'], index=0)
        outlier_trimmer = OutlierTrimmer(
            capping_method=capping_method,
            tail=tail,
            fold=fold,
            variables=cols if cols else None,
            missing_values=missing_values
        )
        return outlier_trimmer.fit_transform(self.dataset)

    def winsorizer(self):
        cols = self.common_parameters()
        capping_method = st.selectbox('Select Winsorization Method', ['gaussian', 'iqr', 'mad', 'quantiles'], index=0)
        tail = st.selectbox('Select Tail for Winsorization', ['left', 'right', 'both'], index=1)
        fold = st.slider('Select Fold Value', 1.0, 5.0, 3.0, 0.1)
        add_indicators = st.checkbox('Add Indicator Variables', value=False)
        missing_values = st.selectbox('Missing Values Handling', ['raise', 'ignore'], index=0)
        winsorizer = Winsorizer(
            capping_method=capping_method,
            tail=tail,
            fold=fold,
            add_indicators=add_indicators,
            variables=cols if cols else None,
            missing_values=missing_values
        )
        return winsorizer.fit_transform(self.dataset)
