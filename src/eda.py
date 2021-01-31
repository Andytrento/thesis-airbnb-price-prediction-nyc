import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
# import tikzplotlib
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import numpy as np

plt.style.use("fivethirtyeight")
# %matplotlib inline

df = pd.read_csv("data/eda.csv")

df.host_since = pd.to_datetime(df.host_since)
df.first_review = pd.to_datetime(df.first_review)

# rename columns

# df.rename(columns={"neighbourhood_cleansed": "neighbourhood"}, inplace=True)
# df.rename(columns={"neighbourhood_group_cleansed": "borough"}, inplace=True)

# Time Series Analysis

# print(
    # f"Of the Airbnb hosts that are still listing on the site, the first joined on {min(df.host_since).strftime('%d %B %Y')}, and the most recent joined on {max(df.host_since).strftime('%d %B %Y')}."
# )
# How long have hosts been listing properties on Airbnb in New York

# plt.style.use("ggplot")
# plt.figure(figsize=(25, 15))
# df.set_index("host_since").resample("MS").size().plot(
    # label="Hosts joining Airbnb", color="green"
# )
# # df.set_index("first_review").resample("MS").size().plot( label="Listings getting their first review", color="green")
# plt.title("New York hosts joining Airbnb")
# plt.legend()
# plt.xlim("2008-08-01", "2019-12-31")  # Limiting to whole months
# plt.xlabel("")
# plt.ylabel("")
# plt.show()


ts_host_since = pd.DataFrame(df.set_index("host_since").resample("MS").size())
ts_first_review = pd.DataFrame(df.set_index("first_review").resample("MS").size())

# Renaming columns
ts_host_since = ts_host_since.rename(columns={0: "hosts"})
ts_host_since.index.rename("month", inplace=True)
ts_first_review = ts_first_review.rename(columns={0: "reviews"})
ts_first_review.index.rename("month", inplace=True)


def decompose_time_series(df, title="", save_name=""):
    """
    Plots the original time series and its decomposition into trend, seasonal and residual.
    """
    # Decomposing the time series
    decomposition = seasonal_decompose(df)

    # Getting the trend, seasonality and noise
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plotting the original time series and the decomposition
    plt.figure(figsize=(12, 8))
    plt.suptitle(title, fontsize=14, y=1)
    plt.subplots_adjust(top=0.80)
    plt.subplot(411)
    plt.plot(df, label="Original")
    plt.legend(loc="upper left")
    plt.subplot(412)
    plt.plot(trend, label="Trend")
    plt.legend(loc="upper left")
    plt.subplot(413)
    plt.plot(seasonal, label="Seasonality")
    plt.legend(loc="upper left")
    plt.subplot(414)
    plt.plot(residual, label="Residuals")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_name)
    # plt.show()


# decompose_time_series(ts_host_since, title="Number of hosts joing Airbnb each month",save_name=
                      # "figures/number-of-host-joining-each-month.pdf")

# decompose_time_series(
# ts_first_review,
# title="Number of Airbnb listings getting their first review each month",
# )


# plt.figure(figsize=(16, 6))
# sns.boxplot(df.host_since.dt.year, np.log(df.host_listings_count))
# plt.xlabel("Year that the host joined Airbnb", fontsize=12)
# plt.ylabel("Number of listings per host (log-transformed)", fontsize=12)
# plt.title(
# "Change per year in the number of listings per host on Airbnb in New York",
# fontsize=16,
# )
# plt.show()


# print("Average number of listings per host per year on Airbnb in Airbnb:")
# print(round(df.set_index("host_since").host_listings_count.resample("YS").mean(), 2))

# List of the largest host_listings_count and the year the host joined Airbnb
# df.sort_values("host_listings_count").drop_duplicates(
# "host_listings_count", keep="last"
# ).tail(10)[["host_since", "host_listings_count"]]

# Question: how have prices changed over time

plt.figure(figsize=(16, 6))
sns.boxplot(df.first_review.dt.year, np.log(df.price))
plt.xlabel("Year that the listing had its first review", fontsize=12)
plt.ylabel("Nightly price (log-transformed)", fontsize=12)
plt.title(
"Change per year in the nightly price of Airbnb listings in New York", fontsize=16
)
plt.show()

# Mean nightly price of listings in each year on Airbnb in London
# print(round(df.set_index("first_review").price.resample("YS").mean(), 2))

# Dropping host_since and first_review as they are no longer needed
df.drop(["host_since", "first_review"], axis=1, inplace=True)

## Numerical Features ###

# print(df.describe())

# Price ##

# What is the overall distribution of prices

# print(
# f"Nightly advertised prices range from {min(df.price)} to {max(df.price)}"
# )  # O to 10000

# Distribution of prices from 0 to 1000

plt.figure(figsize=(25, 15))
df.price.hist(bins=100, range=(0, 1000))
plt.margins(x=0)
plt.axvline(200, color="orange", linestyle="--")
plt.title("Airbnb advertised nightly prices in Airbnb up to $1000", fontsize=35)
plt.xlabel("Price ($)", fontsize=30)
plt.ylabel("Number of listings", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.savefig("figures/price-distribution-up-to-1000.pdf")
# plt.show()

# # Distribution of prices from $200 upwards

plt.figure(figsize=(25, 15))
df.price.hist(bins=100)
plt.margins(x=0)
plt.axvline(200, color="orange", linestyle="--")
plt.title("Airbnb advertised nightly prices in New York up to $200", fontsize=35)
plt.xlabel("Price ($)", fontsize=30)
plt.ylabel("Number of listings", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.savefig("figures/price-distribution-up-to-200.pdf")
# plt.show()

df.loc[df.price <= 10, "price"] = 10  # Replace values under 10 with 10
df.loc[df.price >= 1000, "price"] = 1000  # Replace values over 1000 with 1000

# Host listings count ##

# How many listings do hosts have on average? How many multi-listing hosts are there


# print("Median number of listings per host: ", int(df.host_listings_count.median()))
# print("Mean number of listings per host: ", int(round(df.host_listings_count.mean())))


# print(
# f"{int(round(100*len(df[df.host_listings_count == 1])/len(df)))}% of listings are from hosts with one listing."
# )
# Number of people accommodated, bathrooms, bedrooms and beds ##

fig, ax = plt.subplots(figsize=(25,15))
ax = sns.countplot(x="accommodates", data=df)
ax.set_title("Number of Listings by Number of Accommodates", fontsize=35)
ax.set_xlabel("Number of people a host can accommodate",fontsize=30)
ax.set_ylabel("Count of Listings",fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.savefig("figures/accommodates-countplot.pdf")


fig, ax = plt.subplots(figsize=(25,15))
ax = sns.countplot(x="bathrooms", data=df)
ax.set_title("Number of Listings by Number of Bathrooms", fontsize=35)
ax.set_xlabel("Number of Bathrooms",fontsize=30)
ax.set_ylabel("Count of Listings",fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.savefig("figures/bathrooms-countplot.pdf")


fig, ax = plt.subplots(figsize=(25,15))
ax = sns.countplot(x="bedrooms", data=df)
ax.set_title("Number of Listings by Number of Bedrooms", fontsize=35)
ax.set_xlabel("Number of Bedrooms",fontsize=30)
ax.set_ylabel("Count of Listings",fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.savefig("figures/bedrooms-countplot.pdf")


fig, ax = plt.subplots(figsize=(25,15))
ax = sns.countplot(x="beds", data=df)
ax.set_title("Number of Listings by Number of Beds", fontsize=35)
ax.set_xlabel("Number of Beds",fontsize=30)
ax.set_ylabel("Count of Listings",fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
# plt.savefig("figures/beds-countplot.pdf")

# plt.figure(figsize=(15, 10))
# df.groupby("accommodates").price.median().plot(kind="bar")
# plt.title(
# "Median price of Airbnbs accommodating different number of guests", fontsize=15
# )
# plt.xlabel("Number of guests accommodated", fontsize=15)
# plt.ylabel("Median price ($)", fontsize=15)
# plt.xticks(rotation=0)
# plt.xlim(left=0.5)
# plt.show()


plt.figure(figsize=(25, 15))
df.groupby("accommodates").price.median().plot(kind="bar")
plt.title( "Median price of Airbnbs accommodating different number of guests", fontsize=30)
plt.xlabel("Accommodates", fontsize=30)
plt.ylabel("Median price ($)", fontsize=30)
plt.xticks(rotation=0, fontsize=25)
plt.yticks(fontsize=25)
# plt.savefig("figures/median-price-by-accommodates.pdf")
# plt.show()


plt.figure(figsize=(25, 15))
df.groupby("bathrooms").price.median().plot(kind="bar")
plt.title( "Median price of Airbnbs by Number of Bathrooms", fontsize=30)
plt.xlabel("Number of Bathrooms", fontsize=30)
plt.ylabel("Median price ($)", fontsize=30)
plt.xticks(rotation=0, fontsize=25)
plt.yticks(fontsize=25)
# plt.savefig("figures/median-price-by-bathrooms.pdf")
# plt.show()


plt.figure(figsize=(25, 15))
df.groupby("bedrooms").price.median().plot(kind="bar")
plt.title( "Median price of Airbnbs by Number of Bedrooms", fontsize=30)
plt.xlabel("Number of Bedrooms", fontsize=30)
plt.ylabel("Median price ($)", fontsize=30)
plt.xticks(rotation=0, fontsize=25)
plt.yticks(fontsize=25)
# plt.savefig("figures/median-price-by-bedrooms.pdf")
# plt.show()


plt.figure(figsize=(25, 15))
df.groupby("beds").price.median().plot(kind="bar")
plt.title( "Median price of Airbnbs by Number of Beds", fontsize=30)
plt.xlabel("Number of Beds", fontsize=30)
plt.ylabel("Median price ($)", fontsize=30)
plt.xticks(rotation=0, fontsize=25)
plt.yticks(fontsize=25)
# plt.savefig("figures/median-price-by-beds.pdf")
# plt.show()

# plt.figure(figsize=(15, 10))
# df.groupby("bedrooms").price.median().plot(kind="bar")
# plt.title(
# "Median price of Airbnbs accommodating different number of guests", fontsize=15
# )
# plt.xlabel("Number of bedrooms", fontsize=15)
# plt.ylabel("Median price ($)", fontsize=15)
# plt.xticks(rotation=0)
# plt.xlim(left=0.5)
# plt.show()

# df[["accommodates", "bathrooms", "bedrooms", "beds"]].hist(figsize=(8, 6))

## Categoriacal features ###

# Neighborhodd ##

# Which areas have the most airbnb properties, and which are the mos expensive


# importing the London borough boundary GeoJSON file as a df in geopandas

GEODATAPATH = "data/nyc/neighbourhoods.geojson"
map_df = gpd.read_file(GEODATAPATH)

# map_df.head()
map_df.drop("neighbourhood_group", axis=1, inplace=True)

# Creating a datframe of listing counts and median price by borough
borough_df = pd.DataFrame(df.groupby("neighbourhood").size())
borough_df.rename(columns={0: "number_of_listings"}, inplace=True)
borough_df["median_price"] = df.groupby("neighbourhood").price.median().values

borough_map_df = map_df.set_index("neighbourhood").join(borough_df)  # Joining the df

# Plotting the number of listings in each borough
# fig1, ax1 = plt.subplots(1, figsize=(15, 6))
# borough_map_df.plot(column="number_of_listings", cmap="Blues", ax=ax1)
# ax1.axis("off")
# ax1.set_title("Number of Airbnb listings in each New York borough", fontsize=14)
# sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=9000))
# sm._A = []  # Creates an empty array for the data range
# cbar = fig1.colorbar(sm)
# plt.show()

# fig2, ax2 = plt.subplots(1, figsize=(15, 6))
# borough_map_df.plot(column="median_price", cmap="Blues", ax=ax2)
# ax2.axis("off")
# ax2.set_title(
# "Median price of Airbnb listings in each New York neighbourhood", fontsize=14
# )
# sm = plt.cm.ScalarMappable(
# cmap="Blues",
# norm=plt.Normalize(
# vmin=min(borough_map_df.median_price), vmax=max(borough_map_df.median_price)
# ),
# )
# sm._A = []  # Creates an empty array for the data range
# cbar = fig2.colorbar(sm)
# plt.show()

f, ax = plt.subplots(1, 2, figsize=(18, 8))
borough_map_df["neighbourhood_group"].value_counts().plot.pie( explode=[0, 0.05,
                                                                        0, 0,
                                                                        0],
                                                              autopct="%1.1f%%",
                                                              ax=ax[0],
                                                              shadow=True)
ax[0].set_title("Share of Neighborhood")
ax[0].set_ylabel("Neighborhood Share")
sns.countplot(
"neighbourhood_group",
borough_map_df=borough_map_df,
ax=ax[1],
order=borough_map_df["neighbourhood_group"].value_counts().index,
)
ax[1].set_title("Share of Neighborhood")
plt.show()

# fig, ax = plt.subplots()
borough = df.borough.value_counts()
fig = plt.figure(figsize =(25, 15))
df.borough.value_counts().plot.pie(figsize=(25,15), autopct='%1.1f%%')
plt.title('Share of Borough', fontsize=30)
plt.savefig("figures/borough-pie-chart.pdf")



# df.borough.value_counts().plot.pie(figsize=(25,15), autopct='%1.1f%%')
# plt.axis('equal')
# plt.tight_layout()
# plt.savefig("figures/borough-pie-chart.pdf")
# plt.show()

# Top 10 neighbourhood with most listing

plt.figure(figsize=(25, 15))
clr = (
"blue",
"forestgreen",
"gold",
"red",
"purple",
"cadetblue",
"hotpink",
"orange",
"darksalmon",
"brown",
)
df.neighbourhood.value_counts().sort_values(ascending=False)[:10].sort_values().plot(
kind="barh", color=clr, fontsize=14
)
plt.title("Top 10 Neighborhood with Most Listings", fontsize=15)
plt.savefig("figures/top-10-neighbourhood-with-most-listings.png")
plt.show()

# Borough price distribution
plt.figure(figsize=(25,15))
sns.distplot(df[df.borough=='Manhattan'].price,color='maroon',hist=False,label='Manhattan')
sns.distplot(df[df.borough=='Brooklyn'].price,color='black',hist=False,label='Brooklyn')
sns.distplot(df[df.borough=='Queens'].price,color='green',hist=False,label='Queens')
sns.distplot(df[df.borough=='Bronx'].price,color='orange',hist=False,label='Bronx')
sns.distplot(df[df.borough=='Staten Island'].price,color='blue',hist=False,label='Staten Island')
# plt.title('Kernel Density Estimates of Price in New York Boroughs')
plt.legend(loc="upper right",fontsize=25)
plt.xlabel("Price", fontsize=25)
plt.ylabel("Density", fontsize=25)
plt.xticks(rotation=0, fontsize=25)
plt.yticks(fontsize=25)
plt.xlim(10,1000)
plt.savefig('figures/kernel-density-estimate.png')

# Property and room types


def category_count_plot(col, figsize=(8, 4)):
    """
    Plots a simple bar chart of the total count for each category in the column specified.
    A figure size can optionally be specified.
    """
    plt.figure(figsize=figsize)
    # df[col].value_counts().plot(kind="bar")
    sns.set(font_scale=1.5)
    sns.catplot(col, data=df, kind="count", height=8)
    plt.title(col)
    plt.xticks(rotation=0)
    plt.show()


# 70% of properties are apartments. About 55% of listings are entire homes,
# most of the remainers are private room

# explode = (0, 0.1, 0, 0)

# df["room_type"].value_counts().plot.pie(autopct="%1.1f%%", shadow=True, explode=explode)
# df["property_type"].value_counts().plot.pie(
# autopct="%1.1f%%", shadow=True, explode=(0, 0.1, 0)
# )
# for col in ["property_type", "room_type"]:
# category_count_plot(col, figsize=(4, 3))

# print(df[col].value_counts(normalize=True))


fig, ax = plt.subplots(figsize=(25,15))
ax = df.groupby("room_type").price.median().plot(kind="bar", color=["firebrick", "seagreen", "pink", "#d1f28a"])
ax.set_title(
"Median price of Airbnb  Listing by Room Type ", fontsize=40
)
ax.set_xlabel("Room Type", fontsize=30)
ax.set_ylabel("Median price ($)", fontsize=30)
plt.xticks(rotation=0, fontsize=30)
plt.yticks(rotation=0,fontsize=30)
plt.savefig("figures/median-price-by-room-type.pdf")
plt.show()


fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = df.groupby("property_type").price.median().plot(kind="bar", color=["firebrick", "seagreen", "pink", "#d1f28a"])
plt.xlabel("Property Type")
plt.ylabel("Median price")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.savefig("figures/median-price-by-property-type.pdf")

# Reviews ##

fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = df["review_scores_rating"].value_counts().sort_index(ascending=False).plot(kind="bar", color=["firebrick", "seagreen", "pink", "#d1f28a"])
ax.set_xlabel("Ratings")
ax.set_ylabel("Number of Listings")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.savefig("figures/review-rating-distribution.pdf")

# for col in list(df.columns[df.columns.str.startswith("review_scores") == True]):
# category_count_plot(col, figsize=(5, 3))

# First and last reviews ##

# How long have listings been on the site, and how many listings have been
# reviewed recently

# for col in ["time_since_first_review", "time_since_last_review"]:
# category_count_plot(col)


fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = df.groupby("review_scores_rating").price.median().plot(kind="bar", color=["firebrick", "seagreen", "pink"])
plt.xlabel("Review Score")
plt.ylabel("Median price")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.savefig("figures/median-price-by-reviews-rating.pdf")


fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = df.groupby("time_since_first_review").price.median().plot(kind="bar",
                                                         color=["firebrick",
                                                                "seagreen",
                                                                "pink",
                                                                "orange", "blue"])
plt.xlabel("Time Since First Review")
plt.ylabel("Median price")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.savefig("figures/median-price-by-time-since-first-review.pdf")


fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
ax = df.groupby("time_since_last_review").price.median().plot(kind="bar",
                                                         color=["firebrick",
                                                                "seagreen",
                                                                "pink",
                                                                "orange", "blue"])
plt.xlabel("Time Since Last Review")
plt.ylabel("Median price")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.savefig("figures/median-price-by-time-since-last-review.pdf")


## Boolean features ###


def binary_count_and_price_plot(col, figsize=(8, 3)):
    """
    Plots a simple bar chart of the counts of true and false categories in the column specified,
    next to a bar chart of the median price for each category.
    A figure size can optionally be specified.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(col, fontsize=16, y=1)
    plt.subplots_adjust(
        top=0.80
    )  # So that the suptitle does not overlap with the ax plot titles

    df.groupby(col).size().plot(kind="bar", ax=ax1, color=["firebrick", "seagreen"])
    ax1.set_xticklabels(labels=["false", "true"], rotation=0)
    ax1.set_title("Category count")
    ax1.set_xlabel("")

    df.groupby(col).price.median().plot(
        kind="bar", ax=ax2, color=["firebrick", "seagreen"]
    )
    ax2.set_xticklabels(labels=["false", "true"], rotation=0)
    ax2.set_title("Median price ($)")
    ax2.set_xlabel("")

    plt.show()


# Superhosts ##

# Question: what proportion of Airbnb hosts are superhosts, and is it worth
# being one? (a question often asked by hosts)

# print(df.host_is_superhost.value_counts(normalize=True))
# binary_count_and_price_plot("host_is_superhost")

# Host verification ##
# How many hosts are verified, and is it worth it?

# print(df.host_identity_verified.value_counts(normalize=True))
# binary_count_and_price_plot("host_identity_verified")

# Instant booking
# How many properties are instant bookable(i.e, able to be booked without messaging the host first), and is it worth it?
# print(df.instant_bookable.value_counts(normalize=True))
# binary_count_and_price_plot("instant_bookable")

# Amenities ##

# Which amenities are common , and which increase the price of an Airbnb listing?
for col in df.iloc[:, 28:-2].columns:
    binary_count_and_price_plot(col, figsize=(6, 2))


amenities_list = [
    "air_conditioning",
    "bed_linen",
    "tv",
    "coffee_machine",
    "cooking_basics",
    "white_goods",
    "elevator",
    "child_friendly",
    "parking",
    "host_greeting",
    "internet",
    "long_term_stays",
    "pets_allowed",
    "private_entrance",
    "self_check_in",
]

group_1 = [
    "bed_linen",
    "coffee_machine",
    "cooking_basics",
    "elevator",
    "child_friendly",
    "long_term_stays",
    "private_entrance",
    "self_check_in",
    "pets_allowed"
]

group_2 = ["tv", "white_goods", "internet", "air_conditioning"]

group_3 = ["parking", "host_greeting"]


for col in group_1:
    binary_count_and_price_plot(col)


for col in group_2:
    binary_count_and_price_plot(col)


for col in group_3:
    binary_count_and_price_plot(col)

# Correlation analysis

def multi_collinearity_heatmap(df, figsize=(11, 9)):

    """
    Creates a heatmap of correlations between features in the df. A figure size can optionally be set.
    """

    # Set the style of the visualization
    # sns.set(style="white")
    sns.set_theme(style="white")

    # Create a covariance matrix
    corr = df.corr()

    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    plt.xticks(rotation=90)
    # plt.yticks(rotation=-90)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        vmax=corr[corr != 1.0].max().max(),
    )
    plt.savefig('figures/corr-matrix.pdf')
    plt.show()

# multi_collinearity_heatmap(df, figsize=(11,9))

corr = df.corr()
# print(type(corr))
print(corr.style.background_gradient(cmap='coolwarm').to_latex())
# plt.show()

