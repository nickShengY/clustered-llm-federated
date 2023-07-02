import numpy as np
import tensorflow as tf
import flwr as fl
from datasets import load_dataset
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt

# Load the IMDb dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Define the Flower client
class MyFlowerClient(fl.client.NumPyClient):
    def __init__(self, train_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        self.tokenizer = AutoTokenizer.from_pretrained("models/bert-base-uncased-imdb", use_auth_token="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")
        self.model = TFAutoModelForSequenceClassification.from_pretrained("models/bert-base-uncased-imdb", use_auth_token="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")

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        # Fine-tune the model on the local client dataset
        for example in self.train_dataset:
            text = example["text"]
            label = example["label"]
            inputs = self.tokenizer.encode_plus(text, padding="max_length", truncation=True, return_tensors="tf")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            self.model.train_on_batch(x={"input_ids": input_ids, "attention_mask": attention_mask}, y=np.array([label]))

        return self.model.get_weights(), len(self.train_dataset), {}

# Start the federated learning process
fl.client.start_numpy_client("localhost:8080", client=MyFlowerClient(train_dataset))

# Cluster clients based on model similarity
client_weights = [client.get_parameters() for client in client_models]

# Convert client_weights to a numpy array
client_weights = np.array(client_weights)

# Cluster clients based on model similarity
num_clusters = 3  # Number of clusters
centroids, labels = tf.compat.v1.cluster.kmeans(
    inputs=client_weights,
    num_clusters=num_clusters,
    initial_clusters="kmeans_plus_plus",
    distance_metric="cosine"
)

# Visualize the clusters
for client_id, (model, label) in enumerate(zip(client_weights, labels)):
    plt.scatter(model[0], model[1], color=f"C{label}", label=f"Client {client_id+1}")

# Plot the centroids
for centroid in centroids:
    plt.scatter(centroid[0], centroid[1], color="black", marker="x", s=100)

plt.title("Client Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()