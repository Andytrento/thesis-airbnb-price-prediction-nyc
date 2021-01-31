# Importing required libraries
import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import time


seed(123)
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score

# %matplotlib

# Load data
DATAPATH = "data/nyc/listings.csv"
raw_df = pd.read_csv(DATAPATH)

# print(f"The dataset contains {len(raw_df)} Airbnb listings")  # 50599 listings
pd.set_option("display.max_columns", len(raw_df.columns))  # To view all columns
pd.set_option("display.max_rows", 79)

# print(raw_df.head(10).loc[:,["id", "price" ]].to_latex())

# Dropping initial columns
cols_to_drop = [
    "listing_url",
    "scrape_id",
    "last_scraped",
    "name",
    "summary",
    "space",
    "description",
    "neighborhood_overview",
    "notes",
    "transit",
    "access",
    "interaction",
    "house_rules",
    "thumbnail_url",
    "medium_url",
    "picture_url",
    "xl_picture_url",
    "host_id",
    "host_url",
    "host_name",
    "host_location",
    "host_about",
    "host_thumbnail_url",
    "host_picture_url",
    "host_neighbourhood",
    "host_verifications",
    "calendar_last_scraped",
]

df = raw_df.drop(cols_to_drop, axis=1)

# Other columns can be dropped because they contain a majority of null entries.
# print(df.isna().sum().sort_values(ascending=False)[:6])

# missing_value_count = df.isnull().sum()
# missing_value_percentage = 100 * df.isnull().sum()/len(df)
# missing_value_percentage = missing_value_percentage.round(2)
# missing_value_table = pd.concat([missing_value_count, missing_value_percentage], axis=1)
# missing_value_table = missing_value_table.rename(columns={0: 'count', 1: 'percentage'})
# print(missing_value_table.sort_values(by=['percentage'],ascending=False)[:6].to_latex())


df.drop(
    [
        "host_acceptance_rate",
        "square_feet",
        "weekly_price",
        "monthly_price",
        "license",
        "jurisdiction_names",
    ],
    axis=1,
    inplace=True,
)

df.set_index("id", inplace=True)

# print(df.head(20))

# print(sum((df.host_listings_count == df.host_total_listings_count) == False)) # 563
# df.loc[((df.host_listings_count == df.host_total_listings_count) == False)][:5]

df.drop(
    [
        "host_total_listings_count",
        "calculated_host_listings_count",
        "calculated_host_listings_count_entire_homes",
        "calculated_host_listings_count_private_rooms",
        "calculated_host_listings_count_shared_rooms",
    ],
    axis=1,
    inplace=True,
)

# Drop columns relating to city and country.

lat_long = df[["latitude", "longitude"]]

df.drop(
    [
        "zipcode",
        "latitude",
        "longitude",
        "street",
        "neighbourhood",
        "city",
        "state",
        "market",
        "smart_location",
        "country_code",
        "country",
        "is_location_exact",
    ],
    axis=1,
    inplace=True,
)

# There are multiple columns for minimum and maximum night stays, but the
# two main ones will be used as there are few differences between e.g.
# minimum_nights and minimum_minimum_nights. The latter presumably refers to
# the fact that min/max night stays can vary over the year. The default
# (i.e. most frequently applied) min/max night stay values will be used
# instead.

# print(sum((df.minimum_nights == df.minimum_minimum_nights) == False))  # 2395
df.drop(
    [
        "minimum_minimum_nights",
        "maximum_minimum_nights",
        "minimum_maximum_nights",
        "maximum_maximum_nights",
        "minimum_nights_avg_ntm",
        "maximum_nights_avg_ntm",
    ],
    axis=1,
    inplace=True,
)

# Checking whether boolean and categorical features contain sufficient numbers
# of instances in each category to make them worth including:

# Replacing columns with f/t with 0/1
df.replace({"f": 0, "t": 1}, inplace=True)

# Plotting the distribution of numerical and boolean categories

# df.hist(alpha = 0.3,figsize=(35,30))

# From the above, it can be seen that several columns only contain one category and can be dropped:
df.drop(
    [
        "has_availability",
        "host_has_profile_pic",
        "is_business_travel_ready",
        "require_guest_phone_verification",
        "require_guest_profile_picture",
        "requires_license",
    ],
    axis=1,
    inplace=True,
)

# Cleaning individual columns

## Number of Reviews

# print("Only select active listing (that has number_of_reviews > 0)")

df = df[df.number_of_reviews > 0]

# print("data shape", df.shape)

# print("Experiences_offered")

# print(
# df.experiences_offered.value_counts()
# )  # most listings offer no experience so this feature can be dropped

df.drop("experiences_offered", axis=1, inplace=True)

## Host since

### This is a datetime column, and will be converted into a measure of the
# number of days that a host has been on the platform

df.host_since = pd.to_datetime(df.host_since)  # converting to datetime

df["host_days_active"] = (datetime(2019, 12, 4) - df.host_since).astype(
    "timedelta64[D]"
)  # Calculating the number of days

# print("==============")
# print("host_days_active")

# print("Mean days as host: ", round(df["host_days_active"].mean(), 0))  # 1654
# print("Median days as host: ", round(df["host_days_active"].median(), 0))  # 1662

df.host_days_active.fillna(
    df.host_days_active.median(), inplace=True
)  # Replacing null values with the median

### host_response_time ###

# print("==============")
# print("host_response_time")
# print("Null values: ", df.host_response_time.isna().sum())  #12270
# print(
# f"Proportion: {round((df.host_response_time.isna().sum() / len(df)) * 100, 1)}%"
# )  #30.4%

# Number of rows without a value for host_response_time which have also not
# yet had a review

# print( len( df[ df.loc[:, ["host_response_time",
                           # "first_review"]].isnull().sum(axis=1) == 2 ])) #5388

df.host_response_time.fillna(
    "unknown", inplace=True
)  # fill NaN value with 'unknow'
# print(df.host_response_time.value_counts(normalize=True))

### Host_response_rate ###
# print("==============")
# print("Host Response Rate")

# print("Null values:", df.host_response_rate.isna().sum())  # 12270
# print(
# f"Proportion: {round((df.host_response_rate.isna().sum()/len(df))*100, 1)}%"
# )  # 34.9%

df.host_response_rate = df.host_response_rate.str[:-1].astype(
    "float64"
)  # Removing the % sign from the host_response_rate string and converting to  an integer

# print(
# "Mean host response rate: ", round(df["host_response_rate"].mean(), 0)
# )  # 93.0
# print(
# "Median host response rate: ", round(df["host_response_rate"].median(), 0)
# )  # 100.0
# print(
# f"Proportion of 100% host response rates: {round(((df.host_response_rate == 100.0).sum() / (df.host_response_rate.count()) * 100), 1)}%"
# )  # 70.5 %

df.host_response_rate = pd.cut(
    df.host_response_rate,
    bins=[0, 50, 90, 99, 100],
    labels=["0-49%", "50-89%", "90-99%", "100%"],
    include_lowest=True,
)  # Bin into four categories

df.host_response_rate = df.host_response_rate.astype("str")  # Converting to string

# Replace nulls with 'unknown'
df.host_response_rate.replace("nan", "unknown", inplace=True)

# print("=======")
# print("Host Response Rate Value Count")
# print(df.host_response_rate.value_counts())

### host_is_superhost ###

# Number of rows without a value for multiple host-related columns
# len(df[df.loc[ :,['host_since ', 'host_is_superhost', 'host_listings_count',
# 'host_has_profile_pic', 'host_identity_verified']
# ].isnull().sum(axis=1) == 5]) #248

df.dropna(subset=["host_since"], inplace=True)

### property_type ###


# print("==============")
# print("Property Type")

# print(
# df.property_type.value_counts()
# )  #the categories 'apartment', 'house' and 'other' will be used, as most properties can be classified as either apartments or houses

# Replacing categories that are types of houses or apartments
df.property_type.replace(
    {
        "Townhouse": "House",
        "Serviced apartment": "Apartment",
        "Loft": "Apartment",
        "Bungalow": "House",
        "Cottage": "House",
        "Villa": "House",
        "Tiny house": "House",
        "Earth house": "House",
        "Chalet": "House",
    },
    inplace=True,
)

df.loc[
    ~df.property_type.isin(["House", "Apartment"]), "property_type"
] = "Other"  # Replacing other categories with 'other'

### bathrooms, bedrooms and beds ###

for col in ["bathrooms", "bedrooms", "beds"]:
    df[col].fillna(
        df[col].median(), inplace=True
    )  # Missing values will be replaced with the median

### bed_types ###

# print(df.bed_type.value_counts()) # most listings have the same bed type so this feature can be dropped

## Most listings have the same bed type so this feature can be dropped
df.drop("bed_type", axis=1, inplace=True)

### amenities ###

# print(df.amenities[:1].values) # amenities is a list of additional features in the property, e.g. whether it has a TV or parking

# Creating a set of all possible amentities

amenities_list = list(df.amenities)
amenities_list_string = " ".join(amenities_list)
amenities_list_string = amenities_list_string.replace("{", "")
amenities_list_string = amenities_list_string.replace("}", ",")
amenities_list_string = amenities_list_string.replace('"', "")
amenities_set = [x.strip() for x in amenities_list_string.split(",")]
amenities_set = set(amenities_set)

df.loc[df["amenities"].str.contains("24-hour check-in"), "check_in_24h"] = 1
df.loc[
    df["amenities"].str.contains("Air conditioning|Central air conditioning"),
    "air_conditioning",
] = 1
df.loc[
    df["amenities"].str.contains(
        "Amazon Echo|Apple TV|Game console|Netflix|Projector and screen|Smart TV"
    ),
    "high_end_electronics",
] = 1
df.loc[
    df["amenities"].str.contains("BBQ grill|Fire pit|Propane barbeque"), "bbq"
] = 1
df.loc[df["amenities"].str.contains("Balcony|Patio"), "balcony"] = 1
df.loc[
    df["amenities"].str.contains(
        "Beach view|Beachfront|Lake access|Mountain view|Ski-in/Ski-out|Waterfront"
    ),
    "nature_and_views",
] = 1
df.loc[df["amenities"].str.contains("Bed linens"), "bed_linen"] = 1
df.loc[df["amenities"].str.contains("Breakfast"), "breakfast"] = 1
df.loc[df["amenities"].str.contains("TV"), "tv"] = 1
df.loc[
    df["amenities"].str.contains("Coffee maker|Espresso machine"), "coffee_machine"
] = 1
df.loc[df["amenities"].str.contains("Cooking basics"), "cooking_basics"] = 1
df.loc[df["amenities"].str.contains("Dishwasher|Dryer|Washer"), "white_goods"] = 1
df.loc[df["amenities"].str.contains("Elevator"), "elevator"] = 1
df.loc[df["amenities"].str.contains("Exercise equipment|Gym|gym"), "gym"] = 1
df.loc[
    df["amenities"].str.contains("Family/kid friendly|Children|children"),
    "child_friendly",
] = 1

df.loc[df["amenities"].str.contains("parking"), "parking"] = 1
df.loc[
    df["amenities"].str.contains("Garden|Outdoor|Sun loungers|Terrace"),
    "outdoor_space",
] = 1

df.loc[df["amenities"].str.contains("Host greets you"), "host_greeting"] = 1
df.loc[
    df["amenities"].str.contains("Hot tub|Jetted tub|hot tub|Sauna|Pool|pool"),
    "hot_tub_sauna_or_pool",
] = 1
df.loc[df["amenities"].str.contains("Internet|Pocket wifi|Wifi"), "internet"] = 1
df.loc[
    df["amenities"].str.contains("Long term stays allowed"), "long_term_stays"
] = 1
df.loc[df["amenities"].str.contains("Pets|pet|Cat(s)|Dog(s)"), "pets_allowed"] = 1
df.loc[df["amenities"].str.contains("Private entrance"), "private_entrance"] = 1
df.loc[df["amenities"].str.contains("Safe|Security system"), "secure"] = 1
df.loc[df["amenities"].str.contains("Self check-in"), "self_check_in"] = 1
df.loc[df["amenities"].str.contains("Smoking allowed"), "smoking_allowed"] = 1
df.loc[
    df["amenities"].str.contains("Step-free access|Wheelchair|Accessible"),
    "accessible",
] = 1
df.loc[df["amenities"].str.contains("Suitable for events"), "event_suitable"] = 1

# To reduce the number of features ( to void the curse of dimensionality) is to remove the amenities which add relatively little information
# or are relatively unhelpful in differentiating different listings
# Amenity features where either the true or the false category contains  fewer than 10% of listings will be removed.

cols_to_replace_nulls = df.iloc[:, 41:].columns
df[cols_to_replace_nulls] = df[cols_to_replace_nulls].fillna(0)

# Produces  a list of amentity features where one category (true or false)
# contains fewer than 10% of listings

infrequent_amenities = []
for col in df.iloc[:, 41:].columns:
    if df[col].sum() < len(df) / 10:
        infrequent_amenities.append(col)

# print(infrequent_amenities)

# Dropping infrequne t amenity features
df.drop(infrequent_amenities, axis=1, inplace=True)

# Dropping the original amenity feature
df.drop("amenities", axis=1, inplace=True)

### Price ###

# convert price to an integer
df.price = df.price.str[1:-3]
df.price = df.price.str.replace(",", "")
df.price = df.price.astype("int64")

### security_deposit ###

# As with price, this will be converted to an integer - currently it is a
# string because there is a currency sign.  Having a missing value for
# security deposit is functionally the same as having a
# security deposit of £0, so missing values will be replaced with 0.

# print(df.security_deposit.isna().sum())

df.security_deposit = df.security_deposit.str[1:-3]
df.security_deposit = df.security_deposit.str.replace(",", "")
df.security_deposit.fillna(0, inplace=True)
df.security_deposit = df.security_deposit.astype("int64")

### cleaning_fee ###

# As with price, this will be converted to an integer - currently it is a string
# because there is a currency sign.
# As with security deposit, having a missing value for cleaning fee is
# functionally the same as having a cleaning fee of £0, so missing values will
# be replaced with 0.

# print(df.cleaning_fee.isna().sum())

df.cleaning_fee = df.cleaning_fee.str[1:-3]
df.cleaning_fee = df.cleaning_fee.str.replace(",", "")
df.cleaning_fee.fillna(0, inplace=True)
df.cleaning_fee = df.cleaning_fee.astype("int64")

### extra_people ###

# As with price, this will be converted to an integer - currently it is a
# string because there is a currency sign.
# As with security deposit, having a missing value for extra people is
# functionally the same as having an extra people fee of £0, so missing values
# will be replaced with 0.

df.extra_people = df.extra_people.str[1:-3]
df.extra_people = df.extra_people.str.replace(",", "")
df.extra_people.fillna(0, inplace=True)
df.extra_people = df.extra_people.astype("int64")

### calendar_updated ###

# print("Number of categories: ", df.calendar_updated.nunique()) # 94
# print("\nTop five categories:")
# df.calendar_updated.value_counts()[:5]

df.drop("calendar_updated", axis=1, inplace=True)

# availability
df.drop(
    ["availability_30", "availability_60", "availability_365"], axis=1, inplace=True
)

### first_review and last_review ###

# print(
# f"Null values in 'first_review' : {round(100*df.first_review.isna().sum()/len(df), 1)}%"
# )  # 0%

# print(
# f"Null values in 'review_scores_rating' : {round(100*df.review_scores_rating.isna().sum()/len(df), 1)}%"
# )  # 2.3%

df.first_review = pd.to_datetime(df.first_review)  # Converting to datetime
# Calculating the number of days between the first review and the date the data was scraped
df["time_since_first_review"] = (datetime(2019, 12, 4) - df.first_review).astype(
    "timedelta64[D]"
)

# Distribution of the number of days since first review
# df.time_since_first_review.hist(figsize=(15, 5), bins=30)

def bin_column(col, bins, labels, na_label="unknown"):
    """
    Takes in a column nam, bin cut points and labels, replace the original column with a
    binned version, and replaces nulls with 'unknown' if unspecified
    """

    df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    df[col] = df[col].astype("str")
    df[col].fillna(na_label, inplace=True)

# binning time since first review
bin_column(
    "time_since_first_review",
    bins=[0, 182, 365, 730, 1460, max(df.time_since_first_review)],
    labels=["0-6 months", "6-12 months", "1-2 years", "2-3 years", "4+ years"],
    na_label="no reviews",
)

# last_review

df.last_review = pd.to_datetime(df.last_review)
df["time_since_last_review"] = (datetime(2019, 12, 4) - df.last_review).astype(
    "timedelta64[D]"
)

# Distribution of the number of days since last review
# df.time_since_last_review.hist(figsize=(15,5), bins=30);

# Binning time since last review
bin_column(
    "time_since_last_review",
    bins=[0, 14, 60, 182, 365, max(df.time_since_last_review)],
    labels=["0-2 weeks", "2-8 weeks", "2-6 months", "6-12 months", "1+ year"],
    na_label="no reviews",
)

# Dropping last_review - first_review will be kept for EDA and dropped later
df.drop("last_review", axis=1, inplace=True)

### review ratings columns ###

# keep 1 colum for 'review_scores_rating'

review_scores_cols_to_drop = [
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]

df = df.drop(review_scores_cols_to_drop, axis=1)

# Checking the distributions of the review ratings columns

# variables_to_plot = list(
    # df.columns[df.columns.str.startswith("review_scores") == True]
# )

# fig = plt.figure(figsize=(12,8))
# for i, var_name in enumerate(variables_to_plot):
# ax = fig.add_subplot(3,3,i+1)
# df[var_name].hist(bins=10,ax=ax)
# ax.set_title(var_name)
# fig.tight_layout()
# plt.show()

# Creating a list of all review columns that are scored out of 10
# variables_to_plot.pop(0)

# # Binning for all columns scored out of 10
# for col in variables_to_plot:
    # bin_column(
        # col,
        # bins=[0, 8, 9, 10],
        # labels=["0-8/10", "9/10", "10/10"],
        # na_label="no reviews",
    # )

# Binning column scored out of 100
bin_column(
    "review_scores_rating",
    bins=[0, 80, 95, 100],
    labels=["0-79/100", "80-94/100", "95-100/100"],
    na_label="no reviews",
)

### cancellation_policy ###

# print(df.cancellation_policy.value_counts())

# Replacing categories
df.cancellation_policy.replace(
    {
        "super_strict_30": "strict_14_with_grace_period",
        "super_strict_60": "strict_14_with_grace_period",
        "strict": "strict_14_with_grace_period",
        "luxury_moderate": "moderate",
    },
    inplace=True,
)

# number_of_reviews_ltm and reviews_per_month

df.drop(["number_of_reviews_ltm", "reviews_per_month"], axis=1, inplace=True)

# print(df.head(20))


df.rename(columns={"neighbourhood_cleansed": "neighbourhood"}, inplace=True)
df.rename(columns={"neighbourhood_group_cleansed": "borough"}, inplace=True)

# Export data for exploratory data analysis
df.to_csv("data/eda.csv", index=False,header=True )

# Dropping host_since and first_review as they are no longer needed
df.drop(["host_since", "first_review"], axis=1, inplace=True)

df.loc[df.price <= 10, "price"] = 10  # Replace values under 10 with 10
df.loc[df.price >= 1000, "price"] = 1000  # Replace values over 1000 with 1000


df.to_csv("data/data_cleaned.csv")



