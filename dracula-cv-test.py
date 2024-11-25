# Initialize Ray
import ray
ray.init(ignore_reinit_error=True)

# Import required modules
import numpy as np
import faiss
from langchain_ollama import OllamaEmbeddings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import ollama
from itertools import product

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances

# Ensure NLTK punkt tokenizer is downloaded
nltk.download('punkt', quiet=True)

# Read the dracula.txt file
def demo_string():
    try:
        with open('./dracula.txt', 'r') as file:
            dracula = file.read()
        return dracula
    except FileNotFoundError:
        raise FileNotFoundError("The file 'dracula.txt' was not found in the current directory.")

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


# Define the Ray remote test function
@ray.remote
def run_test(strategy, split_method, chunk_size=10000, max_tokens_per_chunk=1000, use_faiss=False):
    try:
        selector = CriticalVectors(
            strategy=strategy,
            num_clusters='auto',
            chunk_size=chunk_size,
            split_method=split_method,
            max_tokens_per_chunk=max_tokens_per_chunk,
            chunks_per_cluster=1,
            use_faiss=use_faiss
        )

        test_str = demo_string()

        relevant_chunks, first_part, last_part = selector.get_relevant_chunks(test_str)

        context = f"beginning:\n{first_part}\n" + "\n".join(relevant_chunks) + f"\n\nlast part:\n{last_part}"

        prompt = f"""[INST]<<SYS>>RESPOND WITH A `consolidated plot summary` OF THE [context]
\n\n[context] {context} [/context]<</SYS>> RESPOND WITH A `consolidated plot summary` OF THE [context][/INST]"""

        res = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])

        summary = res['message']['content'] if 'message' in res and 'content' in res['message'] else "No summary generated."

        return {
            'strategy': strategy,
            'split_method': split_method,
            'summary': summary
        }

    except Exception as e:
        return {
            'strategy': strategy,
            'split_method': split_method,
            'error': str(e)
        }

# Define test configurations
strategies = ['agglomerative', 'kmeans']
split_methods = ['sentences', 'paragraphs']
test_configs = list(product(strategies, split_methods))

print("Test Configurations:")
for config in test_configs:
    print(f"Strategy: {config[0]}, Split Method: {config[1]}")

# Run all tests in parallel
futures = [run_test.remote(strategy, split_method) for strategy, split_method in test_configs]
results = ray.get(futures)

# Display the results
for result in results:
    strategy = result.get('strategy', 'N/A')
    split_method = result.get('split_method', 'N/A')
    summary = result.get('summary', 'No summary generated.')
    error = result.get('error', None)
    
    print(f"\n---\nStrategy: {strategy}, Split Method: {split_method}")
    if error:
        print(f"Error: {error}")
    else:
        print(f"Summary:\n{summary}\n")


"""
2024-11-24 0X:0X:0X,758	INFO worker.py:1777 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8265 
Test Configurations:
Strategy: agglomerative, Split Method: sentences
Strategy: agglomerative, Split Method: paragraphs
Strategy: kmeans, Split Method: sentences
Strategy: kmeans, Split Method: paragraphs

---
Strategy: agglomerative, Split Method: sentences
Summary:
Here is a consolidated plot summary of the context:

**Summary**

The story revolves around Jonathan Harker, a solicitor who travels to Transylvania to finalize the sale of a property to Count Dracula. Unbeknownst to him, he is walking into a trap set by the vampire.

Upon his return, Harker's wife Mina becomes increasingly concerned as her husband becomes distant and evasive about his experiences in Transylvania. It is revealed that Jonathan has become entangled in a world of supernatural horrors and that he has been forced to flee for his life.

As the story unfolds, it becomes clear that Count Dracula has attached himself to Mina's family through their close relationships with Harker and his friends. The vampire seeks to claim Mina as his own, and the living must band together to stop him.

**Main Plot Points**

* Jonathan Harker visits Transylvania to finalize the sale of a property, but he soon discovers that the property is inhabited by Count Dracula.
* Dracula attacks Harker and sends him away in a coffin, which he believes will keep him safe.
* Unbeknownst to Harker, his wife Mina becomes increasingly entangled in the supernatural world as her husband's experiences are shared with her.
* As Mina learns more about the vampire, she becomes increasingly concerned for her family's safety and well-being.
* The story shifts focus to a group of men (Van Helsing, Quincey Morris, Arthur Holmwood, and John Seward) who band together to stop Dracula and save Mina from his clutches.

**Key Characters**

* Count Dracula: A vampire with supernatural powers
* Jonathan Harker: A solicitor who travels to Transylvania and becomes entangled in the vampire's world.
* Mina Harker (née Murray): Jonathan's wife, who becomes increasingly concerned about her husband's experiences and eventually becomes a key player in the battle against Dracula.
* Van Helsing: A wise and experienced professor who helps the group understand the supernatural forces at play and leads the charge against Dracula.

**Themes**

* The struggle between good and evil
* The power of love and relationships to overcome even the most formidable foes
* The dangers of underestimating or ignoring the supernatural


---
Strategy: agglomerative, Split Method: paragraphs
Summary:
Here is a consolidated plot summary of the story:

The story begins with Jonathan Harker, a solicitor who travels to Transylvania to finalize the sale of a property to Count Dracula. Unbeknownst to Harker, he is entering a world of horror and terror. Upon his return to England, Harker tells his fiancée, Mina, about his experiences in Transylvania.

As Mina becomes increasingly concerned about her future with Lucy Westenra, one of her friends who has fallen under the spell of Count Dracula, a group of men - Abraham Van Helsing, Quincey Morris, Arthur Holmwood, and John Seward - come together to warn Mina about the dangers of vampires.

Meanwhile, in England, a series of strange events occur, including Lucy's transformation into a vampire. The men band together to try and save Lucy, but ultimately fail. They then turn their attention to Count Dracula himself, who is hiding in England.

As the story progresses, it becomes clear that Dracula has arrived in England, preying on the innocent and spreading terror throughout the land. The men work together to stop him, using their knowledge of folklore and supernatural powers to try and defeat the vampire.

Throughout the story, there are multiple plot threads that come together. Harker's experiences in Transylvania serve as a warning about the dangers of Dracula, while Mina's situation serves as a catalyst for the group's efforts to stop the vampire.

Ultimately, it is revealed that Lucy has become a vampire herself, and that she has bitten other people, spreading the curse further. The men are able to destroy her, but not before she has turned several others into vampires.

The story concludes with Dracula being defeated by a combination of forces: Quincey Morris's bravery, Van Helsing's knowledge, and Seward's determination. In the end, Mina is revealed to be Jonathan's true love, and they are reunited.


---
Strategy: kmeans, Split Method: sentences
Summary:
Here is a consolidated plot summary of the provided context:

The story begins with Jonathan Harker, who travels to Transylvania to finalize the sale of a property to Count Dracula. Unbeknownst to Harker, he has fallen into the Count's trap, and Dracula escapes from his castle.

Meanwhile, in London, Harker's fiancée Mina, her friend Lucy, and their companions Lord Godalming, Mr. Morris, and Quincey Morris are trying to uncover the mystery of Dracula's identity and his connection to a recent series of events involving death and transformation in London.

Through a séance, Mina is transported to a ship where she finds herself in a dark, strange place, hearing sounds that seem to be coming from outside. When she returns to her bed, she tells her companions about what she heard, including the sound of men running on the deck and a chain being dropped into the sea.

The group realizes that this must be connected to Count Dracula's escape and decides to follow him across the water. They gather clues, including a ship that is likely to be carrying Dracula, and plan their next move.

At the start of the new day, Van Helsing, who has been studying the situation, reveals his plan to his companions. He believes that they should not actively pursue Count Dracula but instead wait for him to come to them, as he will likely try to escape from a ship into the waters around London.

The group decides to rest and prepare themselves for the next stage of their investigation, with Van Helsing assuring Mina's father and other concerned relatives that they are doing everything in their power to protect his daughter.


---
Strategy: kmeans, Split Method: paragraphs
Summary:
Here is a consolidated plot summary of the text:

The story begins with Jonathan Harker, a young solicitor who travels to Transylvania to finalize the sale of a property to Count Dracula. Upon his arrival, he discovers that the Count is a vampire and barely escapes with his life.

Cut to the present day, where a series of events unfolds involving three main characters: Abraham Van Helsing, a doctor who specializes in supernatural cases; John Seward, an asylum director at Arkham Sanitarium; and Jonathan Harker, who has returned from Transylvania with a severe injury.

Van Helsing reveals that he believes Lucy Westenra, a friend of Harker's, has become a vampire after being bitten by Dracula. He tells Harker that the "proof" for this will be shown in a hospital, where Lucy is being cared for. Van Helsing plans to visit Lucy with Seward and prove the truth about her condition.

The story then jumps forward to the present day, where it is revealed that the original events took place in 1890-1891. Jonathan Harker, John Seward, Abraham Van Helsing, and Mina Murray (a friend of Lucy's) had formed a team to track down and defeat Dracula.

Throughout the story, it is hinted that Dracula has returned from Transylvania and is once again terrorizing the world. The final chapter reveals that the original characters have gone on to lead happy lives: Godalming and Seward are married, and Mina's son will one day understand his mother's bravery and devotion.

However, the story concludes with a sense of foreboding, as it is suggested that Dracula may still be alive and waiting for revenge. The final sentence reads: "We want no proofs; we ask none to believe us! This boy will some day know what a brave and gallant woman his mother is."
"""
