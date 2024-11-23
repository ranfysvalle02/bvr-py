# Initialize Ray
import ray
ray.init(ignore_reinit_error=True)

# Import required modules
import numpy as np
import faiss
from langchain_ollama import OllamaEmbeddings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import requests
import ollama
from sklearn.metrics import pairwise_distances_argmin_min
from itertools import product

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
        valid_strategies = ['kmeans', 'agglomerative', 'map_reduce']
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
        elif self.strategy == 'map_reduce':
            return self._select_chunks_map_reduce(chunks, embeddings, num_clusters)
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
    def _select_chunks_map_reduce(self, chunks, embeddings, num_clusters):
        """
        Selects chunks using a MapReduce-like strategy.

        Parameters:
        - chunks (List[str]): List of text chunks.
        - embeddings (np.ndarray): Embeddings of the chunks.
        - num_clusters (int): Number of clusters.

        Returns:
        - List[str]: Selected chunks.
        """
        # Map Step: Cluster the embeddings
        if self.use_faiss:
            try:
                d = embeddings.shape[1]
                kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=False)
                kmeans.train(embeddings)
                D, I = kmeans.index.search(embeddings, 1)
                labels = I.flatten()
            except Exception as e:
                raise RuntimeError(f"Error in FAISS KMeans clustering during map step: {e}")
        else:
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_clusters, random_state=1337)
                kmeans.fit(embeddings)
                labels = kmeans.labels_
            except Exception as e:
                raise RuntimeError(f"Error in KMeans clustering during map step: {e}")

        # Reduce Step: Select representative chunks from each cluster
        try:
            selected_chunks = []
            for cluster_id in range(num_clusters):
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) == 0:
                    continue  # Skip empty clusters
                cluster_embeddings = embeddings[cluster_indices]
                if self.use_faiss:
                    centroid = np.mean(cluster_embeddings, axis=0).astype('float32').reshape(1, -1)
                    index = faiss.IndexFlatL2(embeddings.shape[1])
                    index.add(cluster_embeddings)
                    D, I = index.search(centroid, 1)
                    closest_index = cluster_indices[I[0][0]]
                else:
                    from sklearn.metrics import pairwise_distances_argmin_min
                    centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
                    closest_idx, _ = pairwise_distances_argmin_min(centroid, cluster_embeddings)
                    closest_index = cluster_indices[closest_idx[0]]
                selected_chunks.append(chunks[closest_index])
            return selected_chunks
        except Exception as e:
            raise RuntimeError(f"Error in Reduce step of MapReduce strategy: {e}")

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
strategies = ['agglomerative', 'kmeans', 'map_reduce']
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
2024-11-23 0X:0X:0X,832	INFO worker.py:1777 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8266 
Test Configurations:
Strategy: agglomerative, Split Method: sentences
Strategy: agglomerative, Split Method: paragraphs
Strategy: kmeans, Split Method: sentences
Strategy: kmeans, Split Method: paragraphs

---
Strategy: agglomerative, Split Method: sentences
Summary:
Here is a consolidated plot summary of the context:

**Main Plot**

The story revolves around the struggles of Jonathan and Mina Harker, a young couple who are trying to deal with supernatural forces that have entered their lives. The narrative unfolds through multiple journals and perspectives, including those of John Harker (Jonathan's brother), Abraham Van Helsing (a Dutch doctor and expert in the supernatural), Quincey Morris (an American friend of Jonathan's), Arthur Holmwood (Quincey's friend and suitor), and Mina herself.

The story begins with John Harker traveling to Transylvania to finalize the sale of a property to Count Dracula, unaware of the horror that awaits him. Upon his return, he confides in his brother and wife about his terrifying experiences, including the death of his servant, Renfield.

As the story progresses, it becomes clear that Mina is also being stalked by Dracula, who has become obsessed with her. The group of friends, led by Van Helsing, band together to stop Dracula's evil plans and save Mina from his clutches.

**Themes and Tone**

Throughout the narrative, the tone shifts between horror, suspense, and a sense of inevitability. The theme of the struggle between good and evil is a dominant one, with the characters facing off against the forces of darkness embodied by Dracula.

The story also explores themes of love, loyalty, and sacrifice, particularly in the relationships between Mina and her friends, as well as Jonathan's devotion to his wife.

**Key Events**

* John Harker travels to Transylvania and discovers the horrors that await him.
* Renfield is killed by Dracula, and his subsequent madness serves as a warning sign for the group.
* Mina is attacked by Dracula, but is saved by Van Helsing and Quincey.
* The group of friends works together to uncover the secrets of Dracula's powers and weaknesses.
* Jonathan becomes increasingly obsessed with stopping Dracula, despite Mina's wishes that he keep her safe.

**Climax and Conclusion**

The climax of the story revolves around the final confrontation between Van Helsing and Dracula. In the end, it is Van Helsing who delivers the fatal blow to the vampire, saving Mina from his grasp. The narrative concludes with a sense of relief and closure, as the surviving characters come to terms with the trauma they have experienced.

Ultimately, the story raises questions about the nature of evil and the power of human love and friendship in the face of darkness and despair.


---
Strategy: agglomerative, Split Method: paragraphs
Summary:
Here is a consolidated plot summary:

The story begins with Jonathan Harker's journey to Transylvania, where he meets Count Dracula and discovers his true nature. Meanwhile, back in England, Harker's fiancée, Mina, becomes ill and is hospitalized. Dr. Seward, a friend of Harker's, takes over as Mina's doctor.

As the story unfolds, it becomes clear that Dracula has arrived in England, spreading vampirism and terrorizing the living. Dr. Seward, along with his assistant Quincey Morris and the solicitor Arthur Holmwood, who is seeking revenge for his sister's death at Dracula's hands, band together to stop him.

The group discovers that Mina has become a vampire herself and is being held by Dracula. They also learn about Dracula's powers and weaknesses, including his aversion to garlic, holy water, and sunlight. Van Helsing, a Dutch doctor who specializes in supernatural diseases, joins the group and helps them develop a plan to defeat Dracula.

The story culminates in a final confrontation between the group and Dracula, with the help of Quincey's death by vampire bite (which allows him to fight back against his undead state), the powers of garlic and holy water, and Van Helsing's knowledge of supernatural lore. In the end, Dracula is defeated, and Mina is restored to her human form.

Throughout the story, themes of love, sacrifice, and the struggle between good and evil are explored, as well as the power of friendship and determination in the face of overwhelming odds.


---
Strategy: kmeans, Split Method: sentences
Summary:
Here is a consolidated plot summary of the context:

The story begins with Jonathan Harker, a young solicitor who travels to Transylvania to finalize the sale of a property to Count Dracula. While there, he becomes trapped in the castle and discovers that he is a prisoner. He manages to escape with the help of a servant named Renfield.

Unbeknownst to Harker, his experiences in the castle have caused him significant physical and emotional distress, leading to his death in England.

Back in England, Harker's fiancée, Mina, begins to experience strange occurrences, including visions and auditory hallucinations. She becomes increasingly ill and is eventually admitted to an insane asylum.

It is revealed that Mina has become the target of Dracula's attentions, and she has begun to see and hear things that are not there - she believes that she can hear the lapping of water and see a ship at sea, even though she is in her bed. She eventually awakens from a trance-like state, unaware of what she experienced.

As Mina recovers, it becomes clear that Dracula's powers have caused her to become possessed by his spirit. The group of men who are concerned for her well-being, including Dr. Van Helsing, Lord Godalming, Mr. Morris, and Quincey Morris (Jonathan's friend), realize that they must find a way to stop Dracula and save Mina.

The group discovers that Dracula is attempting to escape from England by taking his coffin on board a ship in the port of London. They devise a plan to follow him and prevent him from escaping, and they begin to track his movements around the city.

Ultimately, the group is able to find Dracula's ship and follow it to its destination, but the story ends with no further details about what happens next.


---
Strategy: kmeans, Split Method: paragraphs
Summary:
Here is a consolidated plot summary of the context:

**Plot Summary**

The story begins with Jonathan Harker, who travels to Transylvania to finalize the sale of a property to Count Dracula. Unbeknownst to Harker, he has been invited to the castle as a guest, not just a business acquaintance.

Upon his arrival at the castle, Harker discovers that he is a prisoner and that Count Dracula is a vampire. He manages to escape from the castle with the help of a local villager named Quincey Morris.

The story then shifts to London, where Harker's friend, Abraham Van Helsing, a professor of folklore and mythology, learns about Harker's encounter with Dracula. Van Helsing informs Harker that he is being drawn into a supernatural struggle against vampires.

Harker joins forces with Van Helsing, as well as his friends John Seward and Quincey Morris, to help Harker defeat Dracula. Along the way, they discover that Dracula has created a series of vampire minions, including Lucy Westenra, who is transformed into a vampire after being bitten by Dracula.

As the story unfolds, it becomes clear that Dracula's powers are growing stronger, and that he will stop at nothing to spread his evil influence. The group ultimately decides to destroy Dracula's powers and prevent him from turning anyone else into a vampire.

In the final chapters of the book, the protagonists travel to Transylvania to confront Dracula once again. This time, they are joined by Mina Harker, Jonathan's fiancée, who has been drawn into the supernatural struggle against vampires.

The story culminates in a dramatic showdown between the heroes and Dracula, resulting in the vampire's ultimate defeat and destruction.

---
Strategy: map_reduce, Split Method: sentences
Summary:
Here is a consolidated plot summary of the context:

**Summary**

The story begins with Jonathan Harker, a young lawyer who travels to Transylvania to finalize the sale of a property to Count Dracula. Unbeknownst to Harker, he has entered a vampire's lair, and Dracula escapes from his castle while Harker is away.

Meanwhile, back in England, Harker's fiancée, Mina, becomes ill with an unknown disease that seems to be spreading rapidly among the population. Her friend Lucy Westenra also falls victim to the same illness, which eventually leads to her death after a series of bizarre and terrifying events.

The remaining characters - Dr. Seward, Mr. Quincey Morris, Lord Godalming, and Van Helsing - band together to try and stop Dracula's evil plans. They realize that Mina is somehow connected to the supernatural events unfolding in Transylvania and that she may be the key to defeating the vampire.

After a series of encounters with various supernatural entities, including Mina herself, who falls into a trance-like state while trying to communicate with someone, the group discovers that Dracula has taken his lair to England and is hiding on a ship docked in London. They follow the ship's journey, which takes them across the waters, as Mina continues to have visions of the sea.

As they navigate their way through these uncharted territories, the characters receive clues about Dracula's whereabouts, including the Count's escape from his castle and his desire to flee London due to its danger. The group ultimately decides to follow Dracula's trail across the waters, but not before taking some time to rest, eat, and prepare for their journey.

**Themes**

* Supernatural horror and suspense
* Friendship and teamwork in the face of evil
* Love and sacrifice (Jonathan Harker's love for Mina drives his actions throughout the story)
* The power of human connection and understanding between characters

**Notable Characters**

* Count Dracula: the vampire and main antagonist
* Jonathan Harker: the protagonist, a young lawyer who travels to Transylvania and becomes entangled in supernatural events
* Mina Westenra: Harker's fiancée, who becomes ill with an unknown disease and undergoes a series of bizarre transformations
* Dr. Seward: a psychiatrist who houses Lucy Westenra and tries to help her recover from the supernatural illness
* Mr. Quincey Morris: an American adventurer who joins the group to hunt down Dracula
* Lord Godalming: a nobleman who participates in the quest to defeat Dracula
* Van Helsing: a Dutch doctor with expertise in the supernatural, who becomes the team's leader and guide


---
Strategy: map_reduce, Split Method: paragraphs
Summary:
Here is a consolidated plot summary of the context:

A young man named Jonathan Harker travels to Transylvania to finalize the sale of a property to Count Dracula. Upon his return, he shares his terrifying experiences with his friends and acquaintances. One of his friends, Dr. Van Helsing, a Dutch doctor, is revealed to be a vampire hunter.

Van Helsing confides in Harker that Lucy Westenra, a friend of Harker's fiancée Mina, has been transformed into a vampire by Count Dracula through her blood. He believes that the only way to save Lucy is to perform an exorcism and stake through her heart with a cross-stake.

Harker agrees to accompany Van Helsing to visit Lucy in the hospital, where she is being treated for a mysterious illness. However, upon their arrival at the hospital, they discover that Lucy has already been transformed into a vampire, and her blood is spreading rapidly.

Van Helsing's statement "They were made by Miss Lucy" shocks Harker, implying that Lucy created other vampires using her own powers as a vampire. The two men then decide to take action to stop the spread of vampirism and save Lucy.

Note: This summary only covers the initial part of the story and does not include the entire plot, which unfolds over several chapters and involves further adventures with Van Helsing, Mina, and other characters.
"""
