# https://github.com/ranfysvalle02/critical-vectors
import numpy as np
import faiss
from langchain_ollama import OllamaEmbeddings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import requests

# demo setup
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
    A robust class to select the most relevant chunks from a text using various strategies,
    """

    def __init__(
        self,
        chunk_size=500,
        strategy='kmeans',
        num_clusters='auto',
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
            nltk.download('punkt', quiet=True)
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
        Selects the most relevant chunks based on the specified strategy.

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
        Selects chunks using KMeans clustering.

        Parameters:
        - chunks (List[str]): List of text chunks.
        - embeddings (np.ndarray): Embeddings of the chunks.
        - num_clusters (int): Number of clusters.

        Returns:
        - List[str]: Selected chunks.
        """
        if self.use_faiss:
            try:
                d = embeddings.shape[1]
                kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=False)
                kmeans.train(embeddings)
                D, I = kmeans.index.search(embeddings, 1)
                labels = I.flatten()
            except Exception as e:
                raise RuntimeError(f"Error in FAISS KMeans clustering: {e}")
        else:
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_clusters, random_state=1337)
                kmeans.fit(embeddings)
                labels = kmeans.labels_
            except Exception as e:
                raise RuntimeError(f"Error in KMeans clustering: {e}")

        # Find the closest chunk to each cluster centroid
        try:
            if self.use_faiss:
                centroids = kmeans.centroids
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                D, closest_indices = index.search(centroids, 1)
                closest_indices = closest_indices.flatten()
            else:
                from sklearn.metrics import pairwise_distances_argmin_min
                closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
            selected_chunks = [chunks[idx] for idx in closest_indices]
            return selected_chunks
        except Exception as e:
            raise RuntimeError(f"Error selecting chunks: {e}")

    def _select_chunks_agglomerative(self, chunks, embeddings, num_clusters):
        """
        Selects chunks using Agglomerative Clustering.

        Parameters:
        - chunks (List[str]): List of text chunks.
        - embeddings (np.ndarray): Embeddings of the chunks.
        - num_clusters (int): Number of clusters.

        Returns:
        - List[str]: Selected chunks.
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            labels = clustering.fit_predict(embeddings)
        except Exception as e:
            raise RuntimeError(f"Error in Agglomerative Clustering: {e}")

        selected_indices = []
        for label in np.unique(labels):
            cluster_indices = np.where(labels == label)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0).astype('float32').reshape(1, -1)
            # Find the chunk closest to the centroid
            if self.use_faiss:
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(cluster_embeddings)
                D, I = index.search(centroid, 1)
                closest_index_in_cluster = I[0][0]
            else:
                from sklearn.metrics import pairwise_distances_argmin_min
                closest_index_in_cluster, _ = pairwise_distances_argmin_min(centroid, cluster_embeddings)
                closest_index_in_cluster = closest_index_in_cluster[0]
            selected_indices.append(cluster_indices[closest_index_in_cluster])

        selected_chunks = [chunks[idx] for idx in selected_indices]
        return selected_chunks

    def get_relevant_chunks(self, text):
        """
        Gets the most relevant chunks from the text.

        Parameters:
        - text (str): The input text.

        Returns:
        - List[str]: Selected chunks.
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

        # Select the most relevant chunks
        selected_chunks = self.select_chunks(chunks, embeddings)
        return selected_chunks, first_part, last_part

# Example usage:

if __name__ == "__main__":

    # Instantiate the selector
    try:
        selector = CriticalVectors(
            strategy='kmeans',
            num_clusters='auto',
            chunk_size=10000,
            split_method='sentences',
            max_tokens_per_chunk=10000,  # Adjust as needed
            use_faiss=True  # Enable FAISS
        )
        test_str = demo_string()
        # Get the most relevant chunks using the improved method
        relevant_chunks, first_part, last_part = selector.get_relevant_chunks(test_str)
        res = ollama.chat(model=desiredModel, messages=[
            {
                'role': 'user',
                'content': "[INST]<<SYS>>" + "RESPOND WITH A `consolidated plot summary` OF THE [context]" + f"\n\n\[context] beginning:\n{first_part} \n" + "\n".join(relevant_chunks) + f"\n\nlast part:\n{last_part}\n[/context]<</SYS>> RESPOND WITH A `consolidated plot summary` OF THE [context][/INST]",
            },
        ])
        if res['message']:
            print(res['message']['content'])
            exit()
    except Exception as e:
        print(f"An error occurred: {e}")

"""
WARNING clustering 20 points to 5 centroids: please provide at least 195 training points
Here is a consolidated plot summary of the text:

The story begins with Jonathan Harker, a young solicitor who travels to Transylvania to finalize the sale of a property to Count Dracula. Unbeknownst to Harker, he has become entangled in a web of supernatural events. After escaping from the castle, Harker tells his fiancée Mina and her friends about the horrors he experienced.

The group soon discovers that they are all connected to each other through their relationships with Mina, who is also Dracula's victim. They learn that the vampire Quincey Morris was killed by Dracula, but his spirit has passed into their son's body. The group decides to make a journey to Transylvania to revisit the old ground and see if they can find any evidence of the supernatural events.

Upon returning home, the group reviews the papers they have collected, including Harker's journal and Mina's notes. They realize that there is hardly any authentic document among their collection, but Mina's notes and Jonathan's later writings serve as proof of the supernatural events.

The story concludes with a message from Jonathan to his son, who will one day understand what a brave and loving woman his mother is. The group has come to accept the reality of the supernatural world they have encountered, but their experiences have also brought them closer together as friends and family.

Overall, the story is a tale of friendship, love, and survival against the forces of darkness. It follows the journey of Jonathan Harker, Mina, and their companions as they navigate the dangers of vampirism and ultimately find closure and redemption in the face of supernatural adversity.
"""
