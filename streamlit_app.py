import streamlit as st
from streamlit_option_menu import option_menu

class Pipeline:
    def __init__(self, dataset=None):
        self.dataset = dataset
        
        # Mapping stage names to corresponding methods
        self.pipeline_stages = {
            "Missing data imputation": self.missing_data_imputation,
            "Encoding of categorical features": self.encoding_categorical_features,
            "Discretization": self.discretization,
            "Outlier capping or removal": self.outlier_capping_removal,
            "Feature transformation": self.feature_transformation,
            "Creation of new features": self.creation_new_features,
            "Preprocessing": self.preprocessing,
            "Normalization & Scaling": self.normalization_scaling,
            "Regression": self.regression,
            "Classification": self.classification,
            "Clustering": self.clustering
        }
    
    def missing_data_imputation(self, col):
        col.write("### Missing Data Imputation")
        methods = col.multiselect("Select an imputation method:", [
            "MeanMedianImputer", "ArbitraryNumberImputer", "EndTailImputer", "CategoricalImputer", "RandomSampleImputer", "AddMissingIndicator", "DropMissingData"
        ])
        col.write(f"Selected methods: {', '.join(methods)}")
    
    def encoding_categorical_features(self, col):
        col.write("### Encoding of Categorical Features")
        methods = col.multiselect("Select an encoding method:", [
            "OneHotEncoder", "CountFrequencyEncoder", "OrdinalEncoder", "MeanEncoder", "WoEEncoder", "DecisionTreeEncoder", "RareLabelEncoder", "StringSimilarityEncoder",
            "Backward Difference Contrast", "BaseN", "Binary", "Gray", "Count", "Hashing", "Helmert Contrast", "Ordinal", "One-Hot", "Rank Hot", "Polynomial Contrast", "Sum Contrast",
            "CatBoost", "Generalized Linear Mixed Model", "James-Stein Estimator", "LeaveOneOut", "M-estimator", "Target Encoding", "Weight of Evidence", "Quantile Encoder", "Summary Encoder"
        ])
        col.write(f"Selected methods: {', '.join(methods)}")
    
    def discretization(self, col):
        col.write("### Discretization")
        methods = col.multiselect("Select a discretization method:", [
            "EqualFrequencyDiscretiser", "EqualWidthDiscretiser", "ArbitraryDiscretiser", "DecisionTreeDiscretiser", "GeometricWidthDiscretiser"
        ])
        col.write(f"Selected methods: {', '.join(methods)}")
    
    def outlier_capping_removal(self, col):
        col.write("### Outlier Capping or Removal")
        methods = col.multiselect("Select an outlier handling method:", [
            "Winsorizer", "ArbitraryOutlierCapper", "OutlierTrimmer"
        ])
        col.write(f"Selected methods: {', '.join(methods)}")
    
    def feature_transformation(self, col):
        col.write("### Variance Stabilizing Transformations")
        methods = col.multiselect("Select a transformation method:", [
            "LogTransformer", "LogCpTransformer", "ReciprocalTransformer", "ArcsinTransformer", "PowerTransformer", "BoxCoxTransformer", "YeoJohnsonTransformer"
        ])
        col.write(f"Selected methods: {', '.join(methods)}")
    
    def creation_new_features(self, col):
        col.write("### Feature Creation")
        methods = col.multiselect("Select a feature creation method:", [
            "MathFeatures", "RelativeFeatures", "CyclicalFeatures", "DecisionTreeFeatures"
        ])
        col.write(f"Selected methods: {', '.join(methods)}")
    
    def normalization_scaling(self, col):
        col.write("### Normalization & Scaling")
        methods = col.multiselect("Select a scaling method:", [
            "MinMaxScaler", "MaxAbsScaler", "RobustScaler", "StandardScaler"
        ])
        col.write(f"Selected methods: {', '.join(methods)}")
    
    def regression(self, col):
        col.write("### Regression")
        col.write("Running regression analysis...")
    
    def classification(self, col):
        col.write("### Classification")
        col.write("Performing classification...")
    
    def clustering(self, col):
        col.write("### Clustering")
        col.write("Executing clustering algorithms...")
    
    def display(self):
        st.title("Pipeline Application")
        
        # Creating Tabs
        tab1, tab2 = st.tabs(["Pipeline Setup", "Other Features"])
        
        with tab1:
            col1, col2 = st.columns([1, 2])  # Creating two columns in 1:2 ratio
            
            # Radio options for Pipeline
            option = col1.radio("Select an option:", ["Make Pipelines", "Train Pipelines", "Download Models"], index=0)
            
            col2.write("### Details Section")
            
            if option == "Make Pipelines":
                stages = col2.multiselect(
                    "Select the stages for your pipeline:",
                    list(self.pipeline_stages.keys()),
                    format_func=lambda x: x  # Ensures elements appear in one line
                )
                col2.write(f"Selected stages: {', '.join(stages)}")
                
                # Call corresponding methods based on selected stages
                for stage in stages:
                    self.pipeline_stages[stage](col2)
                
# Running the application
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.display()
