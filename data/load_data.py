from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from DANN.data.dataloader import CustomDataset


class MyData:
    def __init__(
        self,
        df: pd.DataFrame,
        class_col=None,
        domain_col=None,
        split_col=None,
        info_include=None,
        batch_size=None,
        control_label=None,
        val_shuffle = True,
    ):
        self.df = df
        self.batch_size = batch_size
        self.info_include = info_include
        self.class_col = class_col
        self.domain_col = domain_col
        self.split_col = split_col
        self.control_label = control_label
        self.val_shuffle = val_shuffle
        
        self.feature_size = None
        self.num_classes = df[class_col].unique().size
        self.num_domains = df[domain_col].unique().size
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        self._imputer = None
        self._label_cat = None
        self._label_encoder = LabelEncoder()
        self._domain_encoder = LabelEncoder()
        
        self._preprocess_data()

    def _preprocess_data(self):
        
        if self.control_label is not None:
            labels = sorted(self.df[self.class_col].unique(), key=lambda x: x != self.control_label)
            self._label_cat = pd.CategoricalDtype(categories=labels, ordered=True)
            self.df[f"{self.class_col}_Encoded"] = self.df[self.class_col].astype(self._label_cat).cat.codes
        else:
            self.df[f"{self.class_col}_Encoded"] = self._label_encoder.fit_transform(
                self.df[f"{self.class_col}_Encoded"]
            )
        self.df[f"{self.domain_col}_Encoded"] = self._domain_encoder.fit_transform(
            self.df[self.domain_col]
        )

        X = self.df.drop(
            [f"{self.class_col}_Encoded", self.class_col, f"{self.domain_col}_Encoded", self.domain_col, self.split_col],
            axis=1,
        )
        if self.info_include is not None:
            X = X.drop(self.info_include, axis=1,errors='ignore')
        y = self.df[f"{self.class_col}_Encoded"]
        d = self.df[f"{self.domain_col}_Encoded"]

        self.train_indices = self.df.loc[self.df[self.split_col].str.contains("train",case=False)].index
        self.val_indices = self.df.loc[self.df[self.split_col].str.contains("valid",case=False)].index
        self.test_indices = self.df.loc[self.df[self.split_col].str.contains("test",case=False)].index

        X_train = X.loc[self.train_indices, :]
        y_train = y[self.train_indices]
        d_train = d[self.train_indices]
        
        X_val = X.loc[self.val_indices, :]
        y_val = y[self.val_indices]
        d_val = d[self.val_indices]
        
        # impute NAs
        self._imputer = SimpleImputer(strategy="mean")
        X_train_imputed = pd.DataFrame(self._imputer.fit_transform(X_train))
        X_val_imputed = pd.DataFrame(self._imputer.transform(X_val))

        # Convert the NumPy arrays to PyTorch tensors
        X_train = torch.FloatTensor(X_train_imputed.values)
        y_train = torch.LongTensor(y_train.values)
        d_train = torch.LongTensor(d_train.values)
        X_val = torch.FloatTensor(X_val_imputed.values)
        y_val = torch.LongTensor(y_val.values)
        d_val = torch.LongTensor(d_val.values)
        
        assert not np.any(np.isnan(X_train_imputed)), "X_train contains NaN values."
        assert not np.any(np.isnan(X_val_imputed)), "X_val contains NaN values."
        
        # data feature
        self.feature_size = X_train.shape[1]
        
        self._train_data = CustomDataset(X_train, y_train, d_train, self.train_indices.to_numpy())
        self._val_data = CustomDataset(X_val, y_val, d_val,self.val_indices.to_numpy())
        
        self.train_loader = DataLoader(
            self._train_data, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(self._val_data, batch_size=self.batch_size,shuffle = self.val_shuffle)
        
        # if has validation set
        if self.test_indices.size > 0:
            X_test = X.loc[self.test_indices, :]
            y_test = y[self.test_indices]
            d_test = d[self.test_indices]
            X_test_imputed = pd.DataFrame(self._imputer.transform(X_test))
            X_test = torch.FloatTensor(X_test_imputed.values)
            y_test = torch.LongTensor(y_test.values)
            d_test = torch.LongTensor(d_test.values)
            assert not np.any(np.isnan(X_test_imputed)), "X_test contains NaN values."
            self._test_data = CustomDataset(X_test, y_test, d_test,self.test_indices.to_numpy())
            self.test_loader = DataLoader(self._test_data, batch_size=self.batch_size)
    
    def load_extradata(
            self,
            new_df: pd.DataFrame,
            class_col2=None,
            info_include=None,
            ):
        if class_col2 is None:
            class_col2 = self.class_col
            
        if self.control_label is not None:
            new_df[f"{class_col2}_Encoded"] = new_df[class_col2].astype(self._label_cat).cat.codes
        else:
            new_df[f"{class_col2}_Encoded"] = self._label_encoder.fit_transform(
                new_df[f"{class_col2}_Encoded"]
            )
        
        X_new = new_df.drop(
            [f"{class_col2}_Encoded", class_col2, f"{self.domain_col}_Encoded", self.domain_col, self.split_col],
            axis=1, errors='ignore'
        )
        if info_include is not None:
            X_new = X_new.drop(info_include, axis=1,errors='ignore')

        y_new = new_df[f"{class_col2}_Encoded"]
        
        # impute NAs
        X_new_impute = pd.DataFrame(self._imputer.transform(X_new))
        
        X_new = torch.FloatTensor(X_new_impute.values)
        y_new = torch.LongTensor(y_new.values)
        d_new = torch.zeros(len(y_new))
        
        assert not np.any(np.isnan(X_new_impute)), "X_new contains NaN values."
        
        new_data = CustomDataset(X_new, y_new, d_new,new_df.index.to_numpy())
        new_loader = DataLoader(new_data, batch_size=self.batch_size)
        
        return new_loader

        