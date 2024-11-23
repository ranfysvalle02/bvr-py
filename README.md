# critical-vectors

---

# Extracting Relevant Text Chunks Using Embeddings and Clustering

![](https://miro.medium.com/v2/resize:fit:1200/0*_XwxbKHayTU8QG44.png)

---

## Introduction

When dealing with large volumes of text, it's often necessary to extract the most relevant or representative parts for tasks like summarization, topic modeling, or information retrieval. Manually sifting through extensive text can be time-consuming and impractical. To address this challenge, **CriticalVectors** was born, a Python class that automates the selection of significant text chunks using embeddings and clustering algorithms.

This tool leverages the power of natural language processing (NLP) and machine learning to extract the most relevant or representative parts from large volumes of text. Here is how it works:

- Split text into manageable chunks.
- Compute embeddings for each chunk to capture semantic meaning.
- Cluster the embeddings to identify groups of similar chunks.
- Select representative chunks from each cluster.

From each cluster, the chunk whose embedding is closest to the centroid (the central point) of that cluster is selected. This chunk is considered the most representative of its cluster. So, if you have N clusters, you will end up with N representative chunks.

## Key Features

- **Flexible Text Splitting**: Split text into sentences or paragraphs based on your preference.
- **Embeddings with Ollama**: Utilize the `OllamaEmbeddings` model with `'nomic-embed-text'` to compute embeddings.
- **Clustering Strategies**: Choose between KMeans or Agglomerative clustering to group similar text chunks.
- **Automatic Cluster Determination**: Automatically determine the optimal number of clusters based on the data.
- **FAISS Integration**: Optionally use Facebook's FAISS library for efficient clustering on large datasets.

You can choose whether to use FAISS by setting the `use_faiss` parameter during initialization. If `use_faiss` is set to `True`, FAISS will be used for clustering. If it's set to `False`, scikit-learn's implementations of the clustering algorithms will be used instead.

## "Lost in the Middle" (sortof)

- **Context Preservation**: While the tool can identify and extract the most representative chunks of text, it might not always preserve the overall narrative or context depending on the length and other factors.
  
## How It Works

### 1. Text Splitting

The input text is split into smaller chunks to make processing manageable. You can choose to split the text by sentences or paragraphs. Additionally, you can specify the maximum number of tokens per chunk to control the size.

```python
chunks = self.split_text(
    text,
    method=self.split_method,
    max_tokens_per_chunk=self.max_tokens_per_chunk
)
```

### 2. Computing Embeddings

Each text chunk is transformed into an embedding vector using the `OllamaEmbeddings` model. This vector representation captures the semantic meaning of the text.

```python
embeddings = self.compute_embeddings(chunks)
```

### 3. Clustering Embeddings

The embeddings are clustered using either KMeans or Agglomerative clustering. Clustering helps group similar chunks together.

```python
if self.strategy == 'kmeans':
    selected_chunks = self._select_chunks_kmeans(chunks, embeddings, num_clusters)
elif self.strategy == 'agglomerative':
    selected_chunks = self._select_chunks_agglomerative(chunks, embeddings, num_clusters)
```

### 4. Selecting Representative Chunks

From each cluster, the chunk closest to the centroid (central point) is selected as the representative. This ensures that the selected chunks are the most representative of their respective clusters.

```python
selected_chunks = [chunks[idx] for idx in closest_indices]
```

## Code Explanation

### Initialization

The `CriticalVectors` class is initialized with several parameters:

- `chunk_size`: The size of each text chunk in characters.
- `strategy`: The clustering strategy (`'kmeans'` or `'agglomerative'`).
- `num_clusters`: The number of clusters or `'auto'` to determine automatically.
- `embeddings_model`: The embeddings model to use. Defaults to `OllamaEmbeddings` with `'nomic-embed-text'`.
- `split_method`: The method to split text (`'sentences'` or `'paragraphs'`).
- `max_tokens_per_chunk`: The maximum number of tokens per chunk.
- `use_faiss`: Whether to use FAISS for clustering.

```python
selector = CriticalVectors(
    strategy='kmeans',
    num_clusters='auto',
    chunk_size=1000,
    split_method='sentences',
    max_tokens_per_chunk=100,
    use_faiss=True
)
```

### Downloading Content

The `demo_string` method fetches text data from a demo URL.

```python
test_str = demo_string()
```

### Splitting Text

The `split_text` method divides the text into chunks based on the specified method.

```python
chunks = self.split_text(
    text,
    method=self.split_method,
    max_tokens_per_chunk=self.max_tokens_per_chunk
)
```

### Computing Embeddings

Embeddings for each chunk are computed using the specified embeddings model.

```python
embeddings = self.compute_embeddings(chunks)
```

### Selecting Chunks

The most relevant chunks are selected based on the clustering strategy.

```python
selected_chunks = self.select_chunks(chunks, embeddings)
```

### Handling Clustering Internals

#### KMeans Clustering

If using KMeans, FAISS can be utilized for efficient computation.

```python
if self.use_faiss:
    kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=False)
    kmeans.train(embeddings)
else:
    kmeans = KMeans(n_clusters=num_clusters, random_state=1337)
    kmeans.fit(embeddings)
```

#### Agglomerative Clustering

For Agglomerative clustering, scikit-learn's implementation is used.

```python
clustering = AgglomerativeClustering(n_clusters=num_clusters)
labels = clustering.fit_predict(embeddings)
```

KMeans and Agglomerative Clustering are two popular types of clustering algorithms. Here's a comparison of the two:

1. **Algorithm Structure:**

   - **KMeans:** KMeans is a centroid-based or partitioning method. It clusters the data into K groups by minimizing the sum of the squared distances (Euclidean distances) between the data and the corresponding centroid. The algorithm iteratively assigns each data point to one of the K groups based on the features that are provided.

   - **Agglomerative Clustering:** Agglomerative Clustering is a hierarchical clustering method. It starts by treating each object as a singleton cluster. Next, pairs of clusters are successively merged until all clusters have been merged into a single cluster that contains all objects. The result is a tree-based representation of the objects, named dendrogram.

2. **Scalability:**

   - **KMeans:** KMeans is more scalable. It can handle large datasets because it only needs to store the centroids of the clusters and the data points are clustered based on their distance to the centroids.

   - **Agglomerative Clustering:** Agglomerative Clustering is less scalable for large datasets. It needs to store the distance matrix which requires a lot of memory for large datasets.

3. **Number of Clusters:**

   - **KMeans:** In KMeans, you need to specify the number of clusters (K) at the beginning.

   - **Agglomerative Clustering:** In Agglomerative Clustering, you don't need to specify the number of clusters at the beginning. You can cut the dendrogram at any level and get the desired number of clusters.

4. **Quality of Clusters:**

   - **KMeans:** KMeans can result in more compact clusters than hierarchical clustering.

   - **Agglomerative Clustering:** Agglomerative Clustering can result in clusters of less compact shape than KMeans.

5. **Use Cases:**

   - **KMeans:** KMeans is useful when you have a good idea of the number of distinct clusters your dataset should be segmented into. Typically, it's good for customer segmentation, document clustering, image segmentation, etc.

   - **Agglomerative Clustering:** Agglomerative Clustering is useful when you don't know how many clusters you might want to end up with. It's good for hierarchical taxonomies, biological taxonomies, etc.

## Comparing Semantic Extraction and Hierarchical Clustering

### Semantic Extraction

Semantic extraction involves understanding and extracting meaningful information from text. This is often done using techniques like Named Entity Recognition (NER), topic modeling, or semantic embeddings.

**When to Use**: Semantic extraction is useful when you want to understand the content of the text, identify key entities or topics, or transform the text into a structured format.

**Pros**:

1. **Understanding Content**: Semantic extraction can help understand the content of the text at a deeper level, identifying key entities, relationships, and topics.

2. **Structured Information**: It can transform unstructured text into a structured format, making it easier to analyze and use in downstream tasks.

3. **Contextual Understanding**: With advanced NLP models, semantic extraction can capture the context and nuances of the text.

**Cons**:

1. **Complexity**: Semantic extraction can be complex and computationally intensive, especially with large volumes of text.

2. **Accuracy**: The accuracy of semantic extraction depends on the quality of the NLP models used. Errors in the models can lead to incorrect extraction.

### Hierarchical Clustering

Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. In the context of text analysis, it can be used to group similar text chunks together based on their semantic similarity.

**When to Use**: Hierarchical clustering is useful when you want to group similar text chunks together, identify themes or topics in the text, or understand the structure and hierarchy of the content.

**Pros**:

1. **Identifying Themes**: Hierarchical clustering can help identify themes or topics in the text, providing a high-level overview of the content.

2. **Understanding Structure**: It can reveal the structure and hierarchy of the content, showing how different parts of the text are related.

3. **No Need for Predefined Number of Clusters**: Unlike other clustering methods like KMeans, hierarchical clustering doesn't require a predefined number of clusters.

**Cons**:

1. **Loss of Context**: Hierarchical clustering might not preserve the original order or context of the text, especially when used for text summarization or extraction.

2. **Computationally Intensive**: Hierarchical clustering can be computationally intensive, especially with large volumes of text and high-dimensional embeddings.

In summary, semantic extraction and hierarchical clustering serve different purposes and have their own strengths and weaknesses. The choice between the two depends on the specific task at hand and the nature of the text being analyzed.

## Potential Applications

- **Text Summarization**: Extract key chunks of text to create a summary.
- **Topic Modeling**: Identify representative chunks for different topics.
- **Information Retrieval**: Quickly retrieve relevant information from large texts.
- **Preprocessing for NLP Tasks**: Prepare data by selecting significant parts.

## Limitations

1. **Context Preservation**: As mentioned earlier, CriticalVectors might not always preserve the overall narrative or context of the text. This is because it selects individual chunks based on their semantic similarity, which might not align with the original order or flow of the text.

2. **Dependence on Embeddings**: The quality of the extracted chunks heavily depends on the quality of the embeddings used. If the embeddings do not capture the semantic meaning of the text well, the selected chunks might not be truly representative.

3. **Computational Resources**: CriticalVectors can be computationally intensive, especially when dealing with large volumes of text and high-dimensional embeddings. This might limit its usability in resource-constrained environments.

4. **Parameter Tuning**: The performance of CriticalVectors can be sensitive to the choice of parameters like `num_clusters`, `max_tokens_per_chunk`, and `chunk_size`. Finding the right set of parameters might require some experimentation and tuning.

5. **Language Support**: The current implementation of CriticalVectors uses the `OllamaEmbeddings` model and NLTK's sentence tokenizer, which might not support all languages. For languages other than English, additional preprocessing and a different embeddings model might be needed.

6. **FAISS Limitations**: While FAISS provides efficient clustering for large datasets, it only supports Euclidean distance and inner product similarity. This might limit its effectiveness for certain types of data or use cases.

7. **Hierarchical Clustering**: Hierarchical clustering can be computationally expensive for large datasets. Also, the dendrogram produced by hierarchical clustering might not always be easy to interpret, especially for large numbers of clusters.

---

## Appendix: Understanding Key Parameters

1. **`max_tokens_per_chunk`**: This parameter sets the maximum number of words, or tokens, that each chunk of text can contain. For instance, if it's set to 100, each chunk will contain up to 100 words. This is particularly important when working with models that have a limit on the number of words they can process at once.

2. **`num_clusters`**: This parameter determines the number of groups, or clusters, into which the chunks of text will be divided during the clustering step. If it's set to 'auto', the number of clusters is calculated automatically based on the number of chunks. The more clusters there are, the more representative chunks will be selected, providing a broader overview of the text.

3. **`chunk_size`**: This parameter sets the maximum size of each chunk in characters when the text is split into paragraphs. For example, if it's set to 1000, each chunk will contain up to 1000 characters. This is useful when you want to control the size of the chunks based on character count rather than word count.

4. **`split_method`**: This parameter determines how the text is divided into chunks. If it's set to 'sentences', the text is split into sentences and then chunks are formed based on `max_tokens_per_chunk`. If it's set to 'paragraphs', the text is split into paragraphs and then chunks are formed based on `chunk_size`.

### How New Chunks Are Created

Depending on the `split_method` chosen, the process of creating new chunks varies:

- If `split_method` is set to 'sentences', the text is first split into individual sentences. These sentences are then grouped into chunks. Once the addition of another sentence would cause a chunk to exceed the `max_tokens_per_chunk` limit, that chunk is completed, and a new chunk is started.

- If `split_method` is set to 'paragraphs', the text is first split into individual paragraphs. These paragraphs are then grouped into chunks. Once the addition of another paragraph would cause a chunk to exceed the `chunk_size` limit, that chunk is completed, and a new chunk is started.

By understanding and appropriately setting these parameters, you can effectively control the granularity and representativeness of the text chunks extracted by the `CriticalVectors` class.

### `max_tokens_per_chunk`

- **Definition**: The maximum number of tokens allowed in each text chunk.
- **Purpose**: Controls the size of each chunk based on token count rather than character count. This is important because models often have token limits.
- **Usage**: When splitting text, sentences are added to a chunk until adding another sentence would exceed the `max_tokens_per_chunk`.
- **Example**: If `max_tokens_per_chunk` is set to `100`, each chunk will contain up to 100 tokens.

```python
max_tokens_per_chunk=100  # Adjust as needed
```

### `num_clusters`

- **Definition**: The number of clusters to form during the clustering step. Can be an integer or `'auto'`.
- **Purpose**: Determines how many representative chunks will be selected.
- **Usage**:
  - If set to an integer, that exact number of clusters will be created.
  - If set to `'auto'`, the number of clusters is determined automatically based on the data.
- **Automatic Calculation**: When `'auto'`, the number of clusters is calculated as:

```python
num_clusters = max(1, int(np.ceil(np.sqrt(num_chunks))))
```

### `chunk_size`

- **Definition**: The size of each text chunk in characters.
- **Purpose**: Controls the maximum size of a chunk when splitting by paragraphs.
- **Usage**: Primarily used when `split_method` is set to `'paragraphs'`. Chunks are formed by combining paragraphs until the `chunk_size` limit is reached.
- **Example**: If `chunk_size` is `1000`, each chunk will contain up to 1000 characters.

```python
chunk_size=1000
```

### `split_method`

- **Definition**: The method used to split the text into chunks. Options are `'sentences'` or `'paragraphs'`.
- **Purpose**: Determines how the text is divided, affecting the granularity of the chunks.
- **Options**:
  - `'sentences'`: Splits the text into sentences and then forms chunks based on `max_tokens_per_chunk`.
  - `'paragraphs'`: Splits the text into paragraphs and then forms chunks based on `chunk_size`.
- **Usage**:

```python
split_method='sentences'  # or 'paragraphs'
```

### Example Usage of Parameters

```python
selector = CriticalVectors(
    strategy='kmeans',
    num_clusters='auto',       # Automatically determine the number of clusters
    chunk_size=1000,           # Maximum size of chunks in characters when using paragraphs
    split_method='sentences',  # Split the text by sentences
    max_tokens_per_chunk=100,  # Maximum number of tokens per chunk
    use_faiss=True             # Use FAISS for efficient clustering
)
```

### Interaction Between Parameters

- When `split_method` is `'sentences'`, the `max_tokens_per_chunk` parameter is used to control chunk sizes.
- When `split_method` is `'paragraphs'`, the `chunk_size` parameter is used instead.
- The `num_clusters` parameter affects how many chunks will be selected as the most relevant.

---

## **CriticalVectors Workflow**

### **1. Initialization**

- **CriticalVectors Instance Creation**
  - **Parameters Set**:
    - `chunk_size`
    - `strategy` (`'kmeans'` or `'agglomerative'`)
    - `num_clusters` (integer or `'auto'`)
    - `embeddings_model` (default is `OllamaEmbeddings`)
    - `split_method` (`'sentences'` or `'paragraphs'`)
    - `max_tokens_per_chunk`
    - `use_faiss` (boolean)
  - **Purpose**: Configures the settings for how text will be processed.

---

### **2. Text Acquisition**

- **Download Raw Content**
  - **Function**: `download_raw_content(url)`
  - **Process**:
    - Fetches text data from the specified URL.
    - Handles exceptions and errors (e.g., HTTP errors).
  - **Outcome**: Raw text data ready for processing.

---

### **3. Text Splitting**

- **Split Text into Chunks**
  - **Method**: `split_text(text, method, max_tokens_per_chunk)`
  - **Options**:
    - **Sentences**:
      - Uses NLTK's sentence tokenizer.
      - Groups sentences into chunks without exceeding `max_tokens_per_chunk`.
    - **Paragraphs**:
      - Splits text based on double newline characters (`'\n\n'`).
  - **Process Flow**:
    1. **Tokenization**: Break text into smaller units (sentences or paragraphs).
    2. **Chunking**: Aggregate tokens into chunks based on size constraints.
  - **Outcome**: A list of text chunks for further processing.

---

### **4. Embedding Computation**

- **Compute Embeddings for Chunks**
  - **Method**: `compute_embeddings(chunks)`
  - **Process**:
    - Converts text chunks into numerical vectors (embeddings).
    - Uses the specified embeddings model (default: `OllamaEmbeddings` with `'nomic-embed-text'`).
    - Ensures embeddings are in `float32` format (required for FAISS).
  - **Outcome**: An array of embeddings representing each chunk in vector space.

---

### **5. Chunk Selection via Clustering**

- **Select Relevant Chunks**
  - **Method**: `select_chunks(chunks, embeddings)`
  - **Strategies**:
    - **KMeans Clustering** (`_select_chunks_kmeans`)
      - Clusters embeddings into `num_clusters`.
      - Uses FAISS or Scikit-learn's KMeans.
      - Selects the chunk closest to each cluster centroid.
    - **Agglomerative Clustering** (`_select_chunks_agglomerative`)
      - Performs hierarchical clustering to group similar chunks.
      - Computes centroids and selects closest chunks.
  - **Automatic Cluster Determination**:
    - If `num_clusters` is `'auto'`, calculates it based on the number of chunks.
  - **Outcome**: A list of the most relevant chunks extracted from the text.

---

### **6. Output Generation**

- **Retrieve Relevant Chunks**
  - **Method**: `get_relevant_chunks(text)`
  - **Process**:
    - Calls `split_text`, `compute_embeddings`, and `select_chunks` in sequence.
    - Also extracts the first and last parts of the text.
  - **Outputs**:
    - `selected_chunks`: The most relevant chunks identified.
    - `first_part`: The initial chunk of the text.
    - `last_part`: The final chunk of the text.

---
