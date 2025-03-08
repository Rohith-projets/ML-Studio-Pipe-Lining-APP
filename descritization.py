import streamlit as st
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser, ArbitraryDiscretiser, DecisionTreeDiscretiser, GeometricWidthDiscretiser

class Discretizers:
    def __init__(self, dataset):
        self.dataset = dataset

    def common_parameters(self):
        # Streamlit widgets for user inputs
        cols = st.multiselect('Select Columns to Discretize', options=self.dataset.columns.tolist())  # Select columns
        return cols

    def equal_frequency_discretiser(self):
        cols = self.common_parameters()
        q = st.slider('Number of Quantiles', 2, 50, 10)  # Quantiles for discretization
        return_object = st.checkbox('Return Object', value=False)  # Checkbox for return object
        return_boundaries = st.checkbox('Return Boundaries', value=False)  # Checkbox for return boundaries
        precision = st.slider('Precision', 1, 10, 3)  # Precision for bin labels
        
        discretizer = EqualFrequencyDiscretiser(
            variables=cols if cols else None,
            q=q,
            return_object=return_object,
            return_boundaries=return_boundaries,
            precision=precision
        )
        return discretizer.fit_transform(self.dataset)

    def equal_width_discretiser(self):
        cols = self.common_parameters()
        bins = st.slider('Number of Bins', 2, 50, 10)  # Number of bins for discretization
        return_object = st.checkbox('Return Object', value=False)
        return_boundaries = st.checkbox('Return Boundaries', value=False)
        precision = st.slider('Precision', 1, 10, 3)
        
        discretizer = EqualWidthDiscretiser(
            variables=cols if cols else None,
            bins=bins,
            return_object=return_object,
            return_boundaries=return_boundaries,
            precision=precision
        )
        return discretizer.fit_transform(self.dataset)

    def arbitrary_discretiser(self):
        cols = self.common_parameters()
        binning_dict = {}
        for col in cols:
            bin_limits = st.text_input(f"Enter bin limits for {col} (comma-separated)", "0,10,100")
            bin_limits = list(map(int, bin_limits.split(','))) if bin_limits else [0, 10, 100]
            binning_dict[col] = bin_limits
        
        return_object = st.checkbox('Return Object', value=False)
        return_boundaries = st.checkbox('Return Boundaries', value=False)
        precision = st.slider('Precision', 1, 10, 3)
        errors = st.selectbox('Handle Errors', ['ignore', 'raise'])
        
        discretizer = ArbitraryDiscretiser(
            binning_dict=binning_dict,
            return_object=return_object,
            return_boundaries=return_boundaries,
            precision=precision,
            errors=errors
        )
        return discretizer.fit_transform(self.dataset)

    def decision_tree_discretiser(self):
        cols = self.common_parameters()
        bin_output = st.selectbox('Select Bin Output', ['prediction', 'bin_number', 'boundaries'], index=0)
        precision = st.slider('Precision', 1, 10, 3)
        cv = st.slider('Cross-Validation Folds', 1, 10, 3)
        scoring = st.selectbox('Scoring Metric', ['neg_mean_squared_error', 'accuracy'], index=0)
        param_grid = st.text_input('Hyperparameter Grid (JSON format)', "{}")
        regression = st.checkbox('Use Regression Tree', value=True)
        
        discretizer = DecisionTreeDiscretiser(
            variables=cols if cols else None,
            bin_output=bin_output,
            precision=precision,
            cv=cv,
            scoring=scoring,
            param_grid=eval(param_grid) if param_grid else None,
            regression=regression
        )
        return discretizer.fit_transform(self.dataset)

    def geometric_width_discretiser(self):
        cols = self.common_parameters()
        bins = st.slider('Number of Bins', 2, 50, 10)
        return_object = st.checkbox('Return Object', value=False)
        return_boundaries = st.checkbox('Return Boundaries', value=False)
        precision = st.slider('Precision', 1, 10, 7)
        
        discretizer = GeometricWidthDiscretiser(
            variables=cols if cols else None,
            bins=bins,
            return_object=return_object,
            return_boundaries=return_boundaries,
            precision=precision
        )
        return discretizer.fit_transform(self.dataset)
