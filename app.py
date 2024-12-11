import os
import pandas as pd
import numpy as np
import re
import nltk
from PIL import Image
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import math
import torch
import torchvision.transforms as transforms
import faiss
import pickle
import csv


# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Folder path where images are stored
image_dir = os.path.expanduser("Images")

# Load the dataset
df = pd.read_csv('semart_train.csv', encoding='latin-1', delimiter='\t')
df['IMAGE_FILE'] = df['IMAGE_FILE'].apply(lambda x: os.path.join(image_dir, x))

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Combine title, technique, and description
df['COMBINED_TEXT'] = df['TITLE'] + " " + df['TECHNIQUE'] + " " + df['DESCRIPTION']
df['COMBINED_TEXT'] = df['COMBINED_TEXT'].apply(preprocess_text)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
image_dir = os.path.expanduser("Images")
df = pd.read_csv('semart_train.csv', encoding='latin-1', delimiter='\t')
df['IMAGE_FILE'] = df['IMAGE_FILE'].apply(lambda x: os.path.join(image_dir, x))

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Combine title, technique, and description
df['COMBINED_TEXT'] = df['TITLE'] + " " + df['TECHNIQUE'] + " " + df['DESCRIPTION']
df['COMBINED_TEXT'] = df['COMBINED_TEXT'].apply(preprocess_text)

# Load a pre-trained model (e.g., ResNet-50)
@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.eval()
    return model

model = load_model()

# Extract features from an image
def extract_features(image):
    try:
        img_tensor = transform(image).unsqueeze(0)
        if img_tensor.shape[1] != 3:
            st.error(f"Uploaded image has {img_tensor.shape[1]} channels, skipping.")
            return None
        with torch.no_grad():
            features = model(img_tensor).squeeze().numpy()
        return features
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Load embeddings and image paths
@st.cache_data
def load_embeddings_and_paths(embeddings_file, paths_file):
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    with open(paths_file, 'r') as csvfile:
        image_paths = [row[0].replace('SemArt/', '') for row in csv.reader(csvfile)]
    return embeddings, image_paths

# Load FAISS index
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(model.fc.out_features)
    index.add(embeddings)
    return index

embeddings, image_paths_loaded = load_embeddings_and_paths('mega_embeddings.pkl', 'image_paths.csv')
faiss_index = create_faiss_index(embeddings)

# Search for similar images
def search_similar_images(query_embedding, index, image_paths, k=5):
    distances, indices = index.search(query_embedding.reshape(1, -1), k=k)
    similar_image_paths = [image_paths[i] for i in indices[0]]
    return similar_image_paths

# TF-IDF Search Function
def search_artworks_tfidf(query, top_n=20):
    query = preprocess_text(query)
    artworks_texts = df['COMBINED_TEXT'].tolist()
    artworks_texts.append(query)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(artworks_texts)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    top_indices = cosine_sim.flatten().argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

# QLM Search Function
def search_artworks_qlm(query, top_n=20, smoothing=1.0):
    query = preprocess_text(query)
    query_tokens = query.split()
    doc_tf = [Counter(preprocess_text(text).split()) for text in df['COMBINED_TEXT']]
    corpus_tf = Counter(word for doc in doc_tf for word in doc)
    corpus_size = sum(corpus_tf.values())
    scores = []
    for doc in doc_tf:
        doc_length = sum(doc.values())
        score = 0
        for term in query_tokens:
            term_frequency = doc.get(term, 0)
            corpus_frequency = corpus_tf.get(term, 0)
            term_prob = (term_frequency + smoothing * (corpus_frequency / corpus_size)) / (doc_length + smoothing)
            score += np.log(term_prob)
        scores.append(score)
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return df.iloc[top_indices]

# BM25 Search Function
def search_artworks_bm25(query, top_n=20, k1=1.2, b=0.75):
    query = preprocess_text(query)
    query_tokens = query.split()
    
    doc_tf = [Counter(preprocess_text(text).split()) for text in df['COMBINED_TEXT']]
    corpus_tf = Counter(word for doc in doc_tf for word in doc)
    doc_lengths = [sum(doc.values()) for doc in doc_tf]
    avg_doc_length = np.mean(doc_lengths)
    scores = []
    N = len(doc_tf)
    
    for doc, doc_length in zip(doc_tf, doc_lengths):
        K = k1 * ((1 - b) + b * (doc_length / avg_doc_length))
        score = 0
        for term in query_tokens:
            fi = doc.get(term, 0)
            ni = sum(1 for doc in doc_tf if term in doc)
            idf = np.log((N - ni + 0.5) / (ni + 0.5) + 1)
            term_weight = ((k1 + 1) * fi) / (K + fi)
            score += idf * term_weight
        scores.append(score)
    
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return df.iloc[top_indices]

# Streamlit App
st.title("Artwork Search Engine")

# Ensure current page is initialized in session state
if "current_page" not in st.session_state:
    st.session_state.current_page = 0

if "previous_query" not in st.session_state:
    st.session_state.previous_query = ""
if "previous_search_type" not in st.session_state:
    st.session_state.previous_search_type = ""

# Search Mode
option = st.radio("Select Search Mode", ("Text Search", "Image Search"))

if option == "Text Search":
    # Search Type
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("Search Type:")
    with col2:
        search_type = st.radio("Search Algorithm", ("TF-IDF", "QLM", "BM25"), horizontal=True)

    # Query Input
    query = st.text_input("Enter your search query:")

    # Reset to page 1 if query or search type changes
    if query != st.session_state.previous_query or search_type != st.session_state.previous_search_type:
        st.session_state.current_page = 0
        st.session_state.previous_query = query
        st.session_state.previous_search_type = search_type

    if query:
        with st.spinner("Searching for artworks..."):
            # Retrieve top 20 results
            if search_type == "TF-IDF":
                results = search_artworks_tfidf(query, top_n=20)
            elif search_type == "QLM":
                results = search_artworks_qlm(query, top_n=20)
            elif search_type == "BM25":
                results = search_artworks_bm25(query, top_n=20)

        # Pagination Variables
        items_per_page = 5
        total_pages = (len(results) + items_per_page - 1) // items_per_page

        # Display Results
        start_idx = st.session_state.current_page * items_per_page
        end_idx = start_idx + items_per_page
        page_results = results.iloc[start_idx:end_idx]

        for _, row in page_results.iterrows():
            img_path = row['IMAGE_FILE']
            try:
                image = Image.open(img_path)
                st.image(image, caption=row['TITLE'], use_container_width =True)
                st.write(f"**Technique:** {row['TECHNIQUE']}")
                st.write(f"**Description:** {row['DESCRIPTION'][:200]}...")

            except Exception as e:
                st.error(f"Error loading image: {e}")

        # Pagination Controls
        st.markdown("---")
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.button("Previous", key="prev") and st.session_state.current_page > 0:
                st.session_state.current_page -= 1
        with col_next:
            if st.button("Next", key="next") and st.session_state.current_page < total_pages - 1:
                st.session_state.current_page += 1

elif option == "Image Search":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Uploaded Image", use_container_width=True)
            query_embedding = extract_features(query_image)
            if query_embedding is not None:
                similar_images = search_similar_images(query_embedding, faiss_index, image_paths_loaded)
                st.write("Similar Images:")
                for img_path in similar_images:
                    img_path_full = os.path.join("Images", img_path)
                    if os.path.exists(img_path_full):
                        try:
                            image = Image.open(img_path_full)
                            st.image(image, caption=img_path, use_container_width=True)

                            # Retrieve metadata from the dataset
                            metadata_row = df[df['IMAGE_FILE'] == img_path_full]
                            if not metadata_row.empty:
                                metadata = metadata_row.iloc[0]
                                st.write(f"**Title:** {metadata['TITLE']}")
                                st.write(f"**Technique:** {metadata['TECHNIQUE']}")
                                st.write(f"**Description:** {metadata['DESCRIPTION'][:200]}...")
                            else:
                                st.warning("Metadata not found for this image.")

                        except Exception as e:
                            st.warning(f"Could not process image: {e}")
                    else:
                        st.warning(f"Image not found: {img_path_full}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

