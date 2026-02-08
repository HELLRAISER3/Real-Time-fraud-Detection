from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd


def get_feature_lists(df):
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    all_num = df.select_dtypes(include=['number']).columns.tolist()
    
    binary_features = []
    numeric_features = []
    
    for col in all_num:
        unique_vals = df[col].dropna().unique()
        
        if len(unique_vals) <= 2:
            binary_features.append(col)
        else:
            numeric_features.append(col)
            
    return cat_features, binary_features, numeric_features


def preprocess_data(df, target='label', test_size=0.15, val_size = 0.1, standardization=False):
    y = df[target]
    X = df.drop(target, axis=1)

    cat_features, binary_features, numeric_features = get_feature_lists(X)
    
    sss = StratifiedShuffleSplit(n_splits=10,
                                 test_size=test_size,
                                 random_state=42)
    
    for i, (train_val_idx, test_idx) in enumerate(sss.split(X,y)):
        X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

    sss = StratifiedShuffleSplit(n_splits=10,
                                 test_size=val_size,
                                 random_state=42)
    
    for i, (train_idx, val_idx) in enumerate(sss.split(X_train_val, y_train_val)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ]
    )    
    if standardization:
        numeric_transformer.steps.append(('scaler', StandardScaler()))
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )
    binary_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', binary_transformer, binary_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='drop' 
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, y_train, X_val_processed, y_val, X_test_processed, y_test






