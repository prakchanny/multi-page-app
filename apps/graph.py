import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.title('Data')

    st.write("This is the `Data` page of the multi-page app.")

    st.write("The following is the DataFrame of the `iris` dataset.")

    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    Y = pd.Series(iris.target, name='class')
    df = pd.concat([X, Y], axis=1)
    df['class'] = df['class'].map({0: "setosa", 1: "versicolor", 2: "virginica"})

    st.write(df)

    st.write("### Distribution of Iris Features")

    # Visualizing the distribution of each feature
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    for idx, feature in enumerate(df.columns[:-1]):
        ax = axes[int(idx / 2), idx % 2]
        sns.histplot(df, x=feature, hue='class', kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature}')

    plt.tight_layout()
    st.pyplot(fig)

if __name__ == '__main__':
    app()

    