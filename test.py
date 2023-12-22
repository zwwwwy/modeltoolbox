import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modeltoolbox.tools as mt
import modeltoolbox as mtb
import seaborn as sns

mt.prefer_settings()

df = pd.read_csv('housing.csv')
op = df['ocean_proximity']
op = pd.get_dummies(op)
df = df.drop('ocean_proximity', axis=1)
df = pd.concat([df, op], axis=1)


def change_col_dtype(DataFrame, before, after):
    """boolcol_to_int.将给定的DateFrame中的所有某(before)类型的列转化为其他(after)类型

    Args:
        DataFrame:
        before: 转化前的类型名
        after: 转化后的类型名
    """
    for column in DataFrame.columns:
        if DataFrame[column].dtype == before:
            DataFrame[column] = DataFrame[column].astype(after)
    return DataFrame


def corr_heatmap(DataFrame, title='pic'):
    """heatmap.快速绘制出一个含有数字的DataFrame的相关系数热力图

    Args:
        DataFrame: pd.DataFrame
        title:
    """
    from seaborn import heatmap
    DataFrame = change_col_dtype(DataFrame, bool, int)
    numeric_columns = DataFrame.select_dtypes(include=['number'])
    heatmap(numeric_columns.corr(), annot=True)
    print(numeric_columns.corr()['median_house_value'].nlargest(4).index.tolist())
    plt.title(title)
    plt.show()


corr_heatmap(df)
# mtb.corr_heatmap(df['median_house_value'])

def fast_corrscatter_evaluate(DataFrame, target, title='pic'):
    from pandas.plotting import scatter_matrix

    DataFrame = change_col_dtype(DataFrame, bool, int)
    numeric_columns = DataFrame.select_dtypes(include=['number'])
    attribute = numeric_columns.corr()[target].nlargest(4).index.tolist()
    scatter_matrix(DataFrame[attribute])
    plt.title(title)
    plt.show()

fast_corrscatter_evaluate(df, 'median_house_value')
