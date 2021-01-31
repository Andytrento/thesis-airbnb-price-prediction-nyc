import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
# seed(123)

df = pd.read_csv('data/data_cleaned.csv')
transformed_df = pd.get_dummies(df)
# Drop collinearity features
to_drop = ['beds',
           'bedrooms',
           'guests_included',
           'host_response_rate_unknown',
           'host_response_rate_0-49%',
           'property_type_Apartment',
           'room_type_Private room']

to_drop.extend(
    list(transformed_df.columns[transformed_df.columns.str.endswith("nan")])
)

transformed_df.drop(to_drop, axis=1, inplace=True)

# Standarizeing and Normalizing

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
    "price",
    "security_deposit",
]

# params = {'axes.titlesize':'32',
          # 'xtick.labelsize':'24',
          # 'ytick.labelsize':'24'}
# mpl.rcParams.update(params)
# transformed_df[numerical_columns].hist(figsize=(20, 15), ec="k")
# plt.tight_layout()

# Other than availability_90 and host_days_active, the remaining numerical
# features are all postively skewed and could benefit from log transformation.
numerical_columns = [
        i for i in numerical_columns if i not in ["availability_90", "host_days_active"]
    ]

for col in numerical_columns:
    transformed_df[col] = transformed_df[col].astype("float64").replace(0.0, 0.01)
    transformed_df[col] = np.log(transformed_df[col])


transformed_df.to_csv('data/prescaled_df.csv', header=True, index=False)

# params = {'axes.titlesize':'15',
          # 'xtick.labelsize':'10',
          # 'ytick.labelsize':'10'}
# mpl.rcParams.update(params)
# transformed_df[numerical_columns].hist(figsize=(20, 15), ec="k")
# plt.tight_layout()

X = transformed_df.drop("price", axis=1)
y = transformed_df.price

# Scaling
# scaler = StandardScaler()

# X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))

# Splitting into train and test sets
# X.to_csv('data/scaled_X.csv', header=True, index=False)
# y.to_csv('data/y.csv', header=True, index=False)

# X_train, X_test, y_train, y_test = train_test_split(
    # X, y, test_size=0.2, random_state=123
# )
# X_train.to_csv('data/X_train.csv', header=True, index=False)
# X_test.to_csv('data/X_test.csv', header=True, index=False)
# y_train.to_csv('data/y_train.csv', header=True, index=False)
# y_test.to_csv('data/y_test.csv', header=True, index=False)






