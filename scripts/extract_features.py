import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

class ApplyVIF:
    def __init__(self, x_train, x_test):
        self.x_train = x_train.copy()
        self.x_test = x_test.copy()
    
    def calculate_vif(self, X):
        vif_data = pd.DataFrame()
        vif_data['Features'] = X.columns
        # Added execution speed optimization: calculate values directly as float64 array matrices
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data

    def apply_vif(self):
        x_train_vif = self.x_train.copy()
        dropped_features = []
        
        while True:
            vif_data = self.calculate_vif(x_train_vif)
            max_vif = vif_data['VIF'].max()
            
            if max_vif > 10:
                drop_feat = vif_data.sort_values('VIF', ascending=False).iloc[0]['Features']
                dropped_features.append(drop_feat)
                print(f"Dropping {drop_feat} with VIF={max_vif:.2f}")
                x_train_vif = x_train_vif.drop(columns=[drop_feat])
            else:
                break

        x_test_vif = self.x_test.drop(columns=dropped_features)
        
        # Explicit print statement to help you capture your 30 final columns for FastAPI validation
        print(f"\nFeature Selection Complete. Retained {len(x_train_vif.columns)} features.")
        return x_train_vif, x_test_vif
