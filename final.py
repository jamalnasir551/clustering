import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def map_corr(df, size=15):
    """Function creates a heatmap of the correlation matrix for each pair of 
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inches)

    The function does not have a plt.show() at the end so that the user 
    can save the figure.
    """
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(size, size))

    cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Plot the heatmap without a mask
    sns.heatmap(corr, cmap=cmap, annot=True, fmt=".2f", linewidths=.5)

    plt.savefig('heatmap.png', dpi=300)


def load_and_preprocess(file):
    """Load the dataset from a CSV file and preprocess it."""
    df = pd.read_csv(file)
    df1 = df.drop(['Country Name', 'Indicator Name'], axis=1)

    # Features to scale
    scaled_columns = df1.columns  # Assuming you want to scale all features

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the selected features
    df1[scaled_columns] = scaler.fit_transform(df1[scaled_columns])

    return df, df1


def elbow_method(df1):
    """Elbow Method to find Optimal k."""
    plt.figure()  # Create a new figure

    inertia = []

    # Perform the Elbow Method for different values of K
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df1)
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow Method
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.savefig('Elbow Method.png', dpi=300)
    plt.show()  # Show the current figure

    return inertia


def apply_kmeans(df, df1, optimal_k=3):
    """Apply K-means clustering and visualize the results."""
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df1)

    common_columns = ['1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000',
                      '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                      '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
                      '2019', '2020']

    # Subset both data frames to include only common columns
    df1_common = df1[common_columns]
    df_common = df[common_columns + ['Country Name', 'Cluster']]

    features = df_common.drop(['Country Name', 'Cluster'], axis=1)
    clusters = df_common['Cluster']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, clusters, test_size=0.2, random_state=42)

    # Check if the training data is not empty
    if X_train.empty:
        print("Training data is empty after splitting.")
        return

    # Train K-means on the training data
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_train)

    # Predict clusters on the test data
    test_clusters = kmeans.predict(X_test)

    # 2D Scatter Plot
    plt.figure()  # Create a new figure
    plt.scatter(X_test['1992'], X_test['1993'], c=test_clusters, cmap='viridis', s=10)  # Adjust 's' value as needed
    plt.title('K-means Clustering - 2D Visualization')
    plt.xlabel('years')  # Replace 'X-axis label' with your desired label for the x-axis
    plt.ylabel('population')  # Replace 'Y-axis label' with your desired label for the y-axis
    plt.savefig('k-means 2-D Plot.png', dpi=300)
    plt.show()  # Show the current figure

    # 3D Scatter Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the actual data points
    ax.scatter(X_test['1992'], X_test['1993'], X_test['1994'], c=test_clusters, cmap='viridis', s=10, label='Data Points')

    # Plotting the cluster centers
    cluster_centers = kmeans.cluster_centers_
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='X', s=100, c='red',
               label='Cluster Centers')

    ax.set_title('K-means Clustering - 3D Visualization')
    ax.set_xlabel('1992')
    ax.set_ylabel('1993')
    ax.set_zlabel('1994')
    plt.legend()
    plt.savefig('k-means 3-D Plot.png', dpi=300)
    plt.show()  # Show the current figure


def silhouette_evaluation(X_test, test_clusters):
    """Evaluate Cohesion and Separation using silhouette score."""
    silhouette = silhouette_score(X_test, test_clusters)
    print(f"Silhouette Score: {silhouette}")


def exponential_growth_fit(selected_country, df):
    """Fit an exponential growth model to CO2 emissions data for a selected country."""
    # Extract relevant rows for the selected country
    country_data = df[df['Country Name'] == selected_country]

    # Extract the years and CO2 emissions values
    years = country_data.columns[2:-1].astype(int)  # assuming '1991' to '2020' columns
    co2_emissions = country_data.iloc[:, 2:-1].values.flatten()

    # Define the exponential growth model
    def exponential_growth(t, a, b):
        return a * np.exp(b * (t - 1991))

    # Provide an initial guess for parameters 'a' and 'b'
    initial_guess = [1.0, 0.01]

    # Fit the model to the data
    params, covariance = curve_fit(exponential_growth, years, co2_emissions)

    # Predict using the fitted parameters
    predicted_values = exponential_growth(years, *params)

    # Plot the original data and the fitted curve
    plt.scatter(years, co2_emissions, label='Actual Data')
    plt.plot(years, predicted_values, color='red', label='Fitted Curve')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions')
    plt.title(f'Exponential Growth Model Fitting for {selected_country}')
    plt.legend()
    plt.savefig('Exponential Growth.png', dpi=300)
    plt.show()

    # Print the fitted parameters
    print("Fitted Parameters (a, b):", params)

    # Calculate confidence intervals
    err_ranges = np.sqrt(np.diag(covariance))
    lower_bound = params - 1.96 * err_ranges
    upper_bound = params + 1.96 * err_ranges

    # Print confidence intervals
    print("Confidence Intervals (95%):")
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)


def main():
    """Main function to execute the analysis."""
    file = 'final.csv'

    # Load and preprocess the data
    df, df1 = load_and_preprocess(file)

    # Visualize correlation matrix
    map_corr(df)

    # Perform the Elbow Method
    inertia = elbow_method(df1)

    # Apply K-means clustering and visualize
    apply_kmeans(df, df1)

    # exponential growth model fitting for a selected country
    selected_country = 'China'
    exponential_growth_fit(selected_country, df)


if __name__ == "__main__":
    main()
