import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def app():
  """
  Streamlit app to perform PCA, Kmeans clustering, and display plot and table.
  """

  # File uploader
  uploaded_file = st.file_uploader("Choose an Excel file")

  if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_excel(uploaded_file)

    # Select numerical columns and include the 'name' column for clustering association
    data_to_cluster = data.select_dtypes(include=[np.number])

    # Convert to numpy array
    data_array = data_to_cluster.to_numpy()

    # Define number of clusters
    k = st.sidebar.number_input("Number of Clusters (K)", min_value=1, value=3)

    # Perform PCA
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data_array)

    # Perform Kmeans clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data_reduced)

    cluster_labels = kmeans.labels_

    # Prepare empty lists to store names in each cluster
    cluster1_names = []
    cluster2_names = []
    cluster3_names = []

    # Loop through data and cluster labels, assign names to corresponding lists
    for i, label in enumerate(cluster_labels):
      if label == 0:
        cluster1_names.append(data['name'].iloc[i])
      elif label == 1:
        cluster2_names.append(data['name'].iloc[i])
      else:
        cluster3_names.append(data['name'].iloc[i])

    # Create the plot
    fig, ax = plt.subplots()
    ax.scatter(data_reduced[:, 0], data_reduced[:, 1], c=cluster_labels)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=150, c='red')
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("K-means Clustering Visualization (PCA)")

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Create a dictionary to store cluster names
    cluster_data = {
      "Cluster 1": cluster1_names,
      "Cluster 2": cluster2_names,
      "Cluster 3": cluster3_names,
    }

    # Ensure all clusters have entries before displaying (adjust for more clusters)
    if len(cluster1_names) > 0 or len(cluster2_names) > 0 or len(cluster3_names) > 0:
      st.write("**Data Points in Each Cluster**")

      # Create a DataFrame with cluster names as the first column and data points as subsequent columns
      cluster_data_df = pd.DataFrame({
        "Cluster": ["Cluster 1", "Cluster 2", "Cluster 3"],
        "Data Points": [", ".join(names) for names in cluster_data.values()]
      })

      st.dataframe(cluster_data_df)
    else:
      st.write("No data points assigned to any cluster yet. Try adjusting the number of clusters (K) or the data.")

    # Display instructions
    st.write("**Instructions**")
    st.write("1. Upload an Excel file containing your data.")
    st.write("2. Set the desired number of clusters (K) in the sidebar.")
    st.write("3. Click 'Run' to visualize the clustered data using PCA and view the table.")

# Run the Streamlit app
if __name__ == "__main__":
  app()