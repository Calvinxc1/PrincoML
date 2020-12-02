import pandas as pd

from .RootNull import RootNull as Root

class DummyNull(Root):
    def init_nulls(self, dataframe):
        dataframe = dataframe.copy()
        null_cols = pd.isnull(dataframe).sum(axis = 0)
        null_cols = null_cols.index[null_cols > 0].values
        self.null_cols = null_cols
        
        for col in self.null_cols:
            null_col = pd.isnull(dataframe[col]).astype(int)
            dataframe.insert(dataframe.columns.get_loc(col)+1, '%s|null' % col, null_col)
            dataframe[col] = dataframe[col].fillna(0)
            
        return dataframe
    
    def col_mapper(self, col):
        if col in self.null_cols:
            return [col, '%s|null' % col]
        else:
            return [col]