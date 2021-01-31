import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns


						for i in range(len(X.columns))]

numerical_columns = [
    "accommodates",
    "availability_90",
    "bathrooms",
    "cleaning_fee",
    "extra_people",
    "host_days_active",
    "host_listings_count",
    "maximum_nights",
    "minimum_nights",
    "number_of_reviews",
    "security_deposit",
]



X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")
y_test = pd.read_csv("data/y_test.csv")
X  = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

numerical_X = X[numerical_columns]
# feat_cols = ["feature" + str(i) for i in range(numerical_X.shape[1])]
# normalised_X = pd.DataFrame(numerical_X, columns=feat_cols)

corr = numerical_X.corr()
sns.heatmap(corr, annot=True)


# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = numerical_X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(numerical_X.values, i) for i in
                   range(len(numerical_X.columns))

print(vif_data.to_latex())

# PCA
pca_airbnb = PCA(n_components=2)
principalComponents_airbnb = pca_airbnb.fit_transform(numerical_X)

principal_airbnb_Df = pd.DataFrame(
    data=principalComponents_airbnb, columns=["PC1", "PC2"]
)

# print(principal_airbnb_Df.tail())
# print(
    # "Explained variation per principal component: {}".format(
        # pca_airbnb.explained_variance_ratio_
    # )
# )

pca = PCA().fit(numerical_X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.savefig('figures/pca.pdf')

def scree_plot():
    from matplotlib.pyplot import figure, show
    from matplotlib.ticker import MaxNLocator

    ax = figure().gca()
    ax.plot(pca.explained_variance_)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, linewidth=1, color='r', alpha=0.5)
    plt.title('Scree Plot of PCA: Component Eigenvalues')
    show()
