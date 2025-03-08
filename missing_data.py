import streamlit as st
from feature_engine.imputation import *

class MissingData:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def MeanMedianImputer(self):
        st.write("### Mean/Median Imputer")
        method = st.radio("Choose imputation method:", ["mean", "median"], index=1)
        variable = st.selectbox("Select variable to impute:", self.dataset.columns.tolist())
        
        if st.button("Apply Imputation", use_container_width=True, type='primary'):
            st.session_state["MeanMedianImputer"] = MeanMedianImputer(imputation_method=method, variables=[variable])
            st.success("Mean/Median Imputer added to session state!")

    def ArbitraryNumberImputer(self):
        st.write("### Arbitrary Number Imputer")
        text_input = st.text_input("Enter arbitrary numbers (comma separated):")
        variables = st.multiselect("Select variable to impute:", self.dataset.columns.tolist())
        
        if st.button("Apply Imputation", use_container_width=True, type='primary'):
            if text_input:
                text_values = text_input.split(',')
                if len(text_values) == len(variables):
                    try:
                        values = [eval(x) for x in text_values]
                        mapper = {variables[i]: values[i] for i in range(len(variables))}
                        st.session_state["ArbitraryNumberImputer"] = ArbitraryNumberImputer(mapper=mapper)
                        st.success("Arbitrary Number Imputer added to session state!")
                    except:
                        st.error("Invalid input values.")
                else:
                    st.error("The number of values should match the number of selected variables.")
            else:
                st.error("Please enter values for imputation.")

    def EndTailImputer(self):
        st.write("### End Tail Imputer")
        imputation_method = st.selectbox("Specify Imputation method", ["gaussian", "iqr", "max"])
        tail = st.radio("Choose tail:", ["right", "left"], index=0)
        fold = st.slider("Select fold value:", 1.0, 5.0, 3.0, step=0.1)
        variables = st.multiselect("Select variables to transform:", self.dataset.columns.tolist())
        
        if st.button("Apply Imputation", use_container_width=True, type='primary'):
            st.session_state["EndTailImputer"] = EndTailImputer(imputation_method=imputation_method, tail=tail, fold=fold, variables=variables)
            st.success("End Tail Imputer added to session state!")

    def CategoricalImputer(self):
        st.write("### Categorical Imputer")
        imputation_method = st.radio("Imputation method:", ["missing", "frequent"], index=0)
        fill_value = st.text_input("Fill value (only used for 'missing' method):", "Missing")
        ignore_format = st.checkbox("Ignore format (apply to all variables)?", value=False)
        return_object = st.checkbox("Return as object?", value=False)
        variables = st.multiselect("Select categorical variables:", self.dataset.select_dtypes(include=['object', 'category']).columns.tolist())
        
        if st.button("Apply Imputation", use_container_width=True, type='primary'):
            st.session_state["CategoricalImputer"] = CategoricalImputer(
                imputation_method=imputation_method, 
                fill_value=fill_value, 
                variables=variables, 
                ignore_format=ignore_format,
                return_object=return_object
            )
            st.success("Categorical Imputer added to session state!")

    def RandomSampleImputer(self):
        st.write("### Random Sample Imputer")
        seed = st.radio("Seeding method:", ["general", "observation"], index=0)
        random_state = st.number_input("Set random state (optional):", min_value=0, value=42, step=1)
        variables = st.multiselect("Select variables to impute:", self.dataset.columns.tolist())

        if st.button("Apply Imputation", use_container_width=True, type='primary'):
            st.session_state["RandomSampleImputer"] = RandomSampleImputer(
                variables=variables, random_state=random_state, seed=seed
            )
            st.success("Random Sample Imputer added to session state!")

    def DropMissingData(self):
        st.write("### Drop Missing Data")
        missing_only = st.radio("Drop rows with missing values only in selected variables?", ["Yes", "No"], index=0) == "Yes"
        threshold = st.slider("Threshold for keeping rows (0-1):", 0.0, 1.0, 1.0, step=0.01)
        variables = st.multiselect("Select variables to check for missing values:", self.dataset.columns.tolist())

        if st.button("Apply Drop Missing Data", use_container_width=True, type='primary'):
            st.session_state["DropMissingData"] = DropMissingData(
                missing_only=missing_only, threshold=threshold, variables=variables
            )
            st.success("Rows with missing values removed from session state!")
