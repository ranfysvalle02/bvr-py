import numpy as np
import faiss
from langchain_ollama import OllamaEmbeddings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import requests
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)

# Demo setup
import ollama
desiredModel = 'llama3.2'
dracula = ""
with open('./dracula.txt', 'r') as file:
    dracula = file.read()

def demo_string():
    return f"""
{dracula}
"""

# Define the CriticalVectors class
class CriticalVectors:
    """
    A robust class to select the most relevant and semantically diverse chunks from a text using various strategies.
    """

    def __init__(
        self,
        chunk_size=500,
        strategy='kmeans',
        num_clusters='auto',
        chunks_per_cluster=1,
        embeddings_model=None,
        split_method='sentences',
        max_tokens_per_chunk=512,
        use_faiss=False
    ):
        """
        Initializes CriticalVectors.

        Parameters:
        - chunk_size (int): Size of each text chunk in characters.
        - strategy (str): Strategy to use for selecting chunks ('kmeans', 'agglomerative').
        - num_clusters (int or 'auto'): Number of clusters (used in clustering strategies). If 'auto', automatically determine the number of clusters.
        - chunks_per_cluster (int): Number of chunks to select per cluster.
        - embeddings_model: Embedding model to use. If None, uses OllamaEmbeddings with 'nomic-embed-text' model.
        - split_method (str): Method to split text ('sentences', 'paragraphs').
        - max_tokens_per_chunk (int): Maximum number of tokens per chunk when splitting.
        - use_faiss (bool): Whether to use FAISS for clustering.
        """
        # Validate chunk_size
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        self.chunk_size = chunk_size

        # Validate strategy
        valid_strategies = ['kmeans', 'agglomerative']
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}.")
        self.strategy = strategy

        # Validate num_clusters
        if num_clusters != 'auto' and (not isinstance(num_clusters, int) or num_clusters <= 0):
            raise ValueError("num_clusters must be a positive integer or 'auto'.")
        self.num_clusters = num_clusters

        # Validate chunks_per_cluster
        if not isinstance(chunks_per_cluster, int) or chunks_per_cluster <= 0:
            raise ValueError("chunks_per_cluster must be a positive integer.")
        self.chunks_per_cluster = chunks_per_cluster

        # Set embeddings_model
        if embeddings_model is None:
            self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        else:
            self.embeddings_model = embeddings_model

        # Set splitting method and max tokens per chunk
        self.split_method = split_method
        self.max_tokens_per_chunk = max_tokens_per_chunk

        # Set FAISS usage
        self.use_faiss = use_faiss

    def split_text(self, text, method='sentences', max_tokens_per_chunk=512):
        """
        Splits the text into chunks based on the specified method.

        Parameters:
        - text (str): The input text to split.
        - method (str): Method to split text ('sentences', 'paragraphs').
        - max_tokens_per_chunk (int): Maximum number of tokens per chunk.

        Returns:
        - List[str]: A list of text chunks.
        """
        # Validate text
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("text must be a non-empty string.")

        if method == 'sentences':
            sentences = sent_tokenize(text)
            chunks = []
            current_chunk = ''
            current_tokens = 0
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                num_tokens = len(tokens)
                if current_tokens + num_tokens <= max_tokens_per_chunk:
                    current_chunk += ' ' + sentence
                    current_tokens += num_tokens
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_tokens = num_tokens
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks
        elif method == 'paragraphs':
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = ''
            for para in paragraphs:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    current_chunk += '\n\n' + para
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks
        else:
            raise ValueError("Invalid method for splitting text.")

    def compute_embeddings(self, chunks):
        """
        Computes embeddings for each chunk.

        Parameters:
        - chunks (List[str]): List of text chunks.

        Returns:
        - np.ndarray: Embeddings of the chunks.
        """
        # Validate chunks
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks must be a non-empty list of strings.")

        try:
            embeddings = self.embeddings_model.embed_documents(chunks)
            embeddings = np.array(embeddings).astype('float32')  # FAISS requires float32
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error computing embeddings: {e}")

    def select_chunks(self, chunks, embeddings):
        """
        Selects the most relevant and semantically diverse chunks based on the specified strategy.

        Parameters:
        - chunks (List[str]): List of text chunks.
        - embeddings (np.ndarray): Embeddings of the chunks.

        Returns:
        - List[str]: Selected chunks.
        """
        num_chunks = len(chunks)
        num_clusters = self.num_clusters

        # Automatically determine number of clusters if set to 'auto'
        if num_clusters == 'auto':
            num_clusters = max(1, int(np.ceil(np.sqrt(num_chunks))))
        else:
            num_clusters = min(num_clusters, num_chunks)

        if self.strategy == 'kmeans':
            return self._select_chunks_kmeans(chunks, embeddings, num_clusters)
        elif self.strategy == 'agglomerative':
            return self._select_chunks_agglomerative(chunks, embeddings, num_clusters)
        else:
            # This should not happen due to validation in __init__
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _select_chunks_kmeans(self, chunks, embeddings, num_clusters):
        """
        Selects chunks using KMeans clustering with semantic diversity.

        Parameters:
        - chunks (List[str]): List of text chunks.
        - embeddings (np.ndarray): Embeddings of the chunks.
        - num_clusters (int): Number of clusters.

        Returns:
        - List[str]: Selected chunks.
        """
        selected_indices = []

        if self.use_faiss:
            try:
                d = embeddings.shape[1]
                kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=False)
                kmeans.train(embeddings)
                centroids = kmeans.centroids
                index = faiss.IndexFlatL2(d)
                index.add(embeddings)
                D, I = index.search(centroids, self.chunks_per_cluster)
                for cluster_idx in range(num_clusters):
                    cluster_chunk_indices = I[cluster_idx]
                    for idx in cluster_chunk_indices:
                        selected_indices.append(idx)
            except Exception as e:
                raise RuntimeError(f"Error in FAISS KMeans clustering: {e}")
        else:
            try:
                kmeans = KMeans(n_clusters=num_clusters, random_state=1337)
                kmeans.fit(embeddings)
                labels = kmeans.labels_
                centroids = kmeans.cluster_centers_
            except Exception as e:
                raise RuntimeError(f"Error in KMeans clustering: {e}")

            try:
                for cluster_idx in range(num_clusters):
                    cluster_indices = np.where(labels == cluster_idx)[0]
                    if len(cluster_indices) == 0:
                        continue
                    cluster_embeddings = embeddings[cluster_indices]
                    centroid = centroids[cluster_idx].reshape(1, -1)
                    # Compute distances to centroid
                    distances = pairwise_distances(cluster_embeddings, centroid, metric='euclidean').flatten()
                    # Sort indices by distance to centroid
                    sorted_indices = cluster_indices[np.argsort(distances)]
                    # Select the closest chunk first
                    selected = [sorted_indices[0]]
                    # Select additional chunks at extremes
                    if self.chunks_per_cluster > 1 and len(sorted_indices) > 1:
                        selected.append(sorted_indices[-1])
                    # If more chunks are needed
                    while len(selected) < self.chunks_per_cluster and len(sorted_indices) > len(selected):
                        # Select the next farthest chunk
                        next_idx = sorted_indices[-len(selected)-1]
                        if next_idx not in selected:
                            selected.append(next_idx)
                    selected_indices.extend(selected)
            except Exception as e:
                raise RuntimeError(f"Error selecting chunks: {e}")

        # Remove duplicate indices
        selected_indices = list(dict.fromkeys(selected_indices))
        selected_chunks = [chunks[idx] for idx in selected_indices]
        return selected_chunks

    def _select_chunks_agglomerative(self, chunks, embeddings, num_clusters):
        """
        Selects chunks using Agglomerative Clustering with semantic diversity.

        Parameters:
        - chunks (List[str]): List of text chunks.
        - embeddings (np.ndarray): Embeddings of the chunks.
        - num_clusters (int): Number of clusters.

        Returns:
        - List[str]: Selected chunks.
        """
        selected_indices = []
        try:
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            labels = clustering.fit_predict(embeddings)
        except Exception as e:
            raise RuntimeError(f"Error in Agglomerative Clustering: {e}")

        try:
            for cluster_idx in range(num_clusters):
                cluster_indices = np.where(labels == cluster_idx)[0]
                if len(cluster_indices) == 0:
                    continue
                cluster_embeddings = embeddings[cluster_indices]
                centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
                # Compute distances to centroid
                distances = pairwise_distances(cluster_embeddings, centroid, metric='euclidean').flatten()
                # Sort indices by distance to centroid
                sorted_indices = cluster_indices[np.argsort(distances)]
                # Select the closest chunk first
                selected = [sorted_indices[0]]
                # Select additional chunks at extremes
                if self.chunks_per_cluster > 1 and len(sorted_indices) > 1:
                    selected.append(sorted_indices[-1])
                # If more chunks are needed
                while len(selected) < self.chunks_per_cluster and len(sorted_indices) > len(selected):
                    # Select the next farthest chunk
                    next_idx = sorted_indices[-len(selected)-1]
                    if next_idx not in selected:
                        selected.append(next_idx)
                selected_indices.extend(selected)
        except Exception as e:
            raise RuntimeError(f"Error selecting chunks: {e}")

        # Remove duplicate indices
        selected_indices = list(dict.fromkeys(selected_indices))
        selected_chunks = [chunks[idx] for idx in selected_indices]
        return selected_chunks

    def get_relevant_chunks(self, text):
        """
        Gets the most relevant and semantically diverse chunks from the text.

        Parameters:
        - text (str): The input text.

        Returns:
        - List[str], str, str: Selected chunks, first part, and last part.
        """
        # Split the text into chunks
        chunks = self.split_text(
            text,
            method=self.split_method,
            max_tokens_per_chunk=self.max_tokens_per_chunk
        )

        if not chunks:
            return [], '', ''

        # first part
        first_part = chunks[0]
        # last part
        last_part = chunks[-1]

        # Compute embeddings for each chunk
        embeddings = self.compute_embeddings(chunks)

        # Select the most relevant and diverse chunks
        selected_chunks = self.select_chunks(chunks, embeddings)
        return selected_chunks, first_part, last_part

# Example usage:

if __name__ == "__main__":
    # Instantiate the selector with configurable chunks_per_cluster parameter
    try:
        selector = CriticalVectors(
            strategy='kmeans',
            num_clusters='auto',
            chunk_size=10000,
            chunks_per_cluster=1,  # Set the desired number of chunks per cluster here
            split_method='sentences',
            max_tokens_per_chunk=3000,  # Adjust as needed
            use_faiss=True  # Enable FAISS if desired
        )
        test_str = demo_string()
        # Get the most relevant and diverse chunks using the improved method
        relevant_chunks, first_part, last_part = selector.get_relevant_chunks(test_str)
        res = ollama.chat(model=desiredModel, messages=[
            {
                'role': 'user',
                'content': "[INST]<<SYS>>" + "RESPOND WITH A `consolidated plot summary` OF THE [context]" +
                           f"\n\n\[context] beginning:\n{first_part} \n" +
                           "\n".join(relevant_chunks) +
                           f"\n\nlast part:\n{last_part}\n[/context]<</SYS>> RESPOND WITH A `consolidated plot summary` OF THE [context][/INST]",
            },
        ])
        if res['message']:
            print(res['message']['content'])
            exit()
    except Exception as e:
        print(f"An error occurred: {e}")

"""
WARNING clustering 64 points to 8 centroids: please provide at least 312 training points
Here is a consolidated plot summary of the story:

**The Story Begins**

Jonathan Harker, a young solicitor, travels to Transylvania to finalize the sale of a property to Count Dracula. Unbeknownst to Harker, the Count is a vampire.

**Harker's Encounter with the Vampire**

Upon arriving at Castle Dracula, Harker discovers that he has been invited to dinner, where he meets the Count. The two men engage in conversation, and Harker soon realizes that the Count is a supernatural being who feeds on human blood.

**Escape from Transylvania**

Harker barely escapes from the castle with his life, only to find himself stranded in England after losing his ticket and identification.

**The Infection of Mina**

Meanwhile, in England, Mina Murray becomes engaged to Harker's friend, Jonathan Harker. Unbeknownst to her, she has been bitten by a vampire (later revealed to be Renfield) and is slowly becoming infected with the curse.

**Quincey Morris' Arrival**

The story then shifts to Texas, where Quincey Morris, an American adventurer, joins forces with Harker's friends to search for him. They soon discover that Mina has been bitten by a vampire.

**The Journey to Transylvania**

The group sets out on a perilous journey to Transylvania to find the Count and save Mina from his curse.

**The Confrontation at the Castle**

Upon arriving at Castle Dracula, the group discovers that Harker had discovered the true nature of the Count and had attempted to stop him. However, they arrive too late, and Harker is killed by the vampire's hand.

**The Final Battle**

In a climactic confrontation, Jonathan Harker and Quincey Morris attempt to destroy the Count, but he proves to be a formidable foe. Just as all hope seems lost, the two men are able to defeat him with the help of stakes and garlic.

**The Aftermath**

After defeating the Count, the group mourns the loss of their friend Quincey Morris, who has been fatally wounded during the battle. As they prepare to leave Transylvania, they realize that the curse has been broken, and Mina is finally free from its grasp.

**The Epilogue**

Seven years later, the story jumps forward in time, where we find out that Harker's death was not in vain. The group has since reunited with their loved ones, including Mina, who has given birth to a healthy child. They celebrate the boy's birthday on the same day as Quincey Morris' death, marking a new beginning for them.

Overall, the story is a thrilling tale of friendship, sacrifice, and the ultimate defeat of evil forces that threaten humanity.
"""
