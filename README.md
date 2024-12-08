# Multi-Modal Information Retrieval with SemArt Dataset

## Overview
This project implements a **multi-modal information retrieval system** using the **SemArt dataset**, which contains fine-art paintings enriched with textual descriptions and associated visual attributes. The system allows users to query artwork using either **textual descriptions** or **images**, and it retrieves relevant artworks based on either textual or visual similarity. 

## Features
- **Text-Based Retrieval:** Given a text query, the system returns artworks whose descriptions are semantically similar to the query.
- **Image-Based Retrieval:** Given an image query, the system retrieves artworks that visually resemble the input image.
- **Ranked Results:** The retrieved artworks are ranked in order of relevance, with the most similar artworks appearing at the top.

## Setup

### 1. Clone the Repository
Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/mh-mubashir/multi-modal-retrieval-system-SemArt.git
cd multi-modal-retrieval-semart
```

### 2. Download the SemArt Dataset
The **SemArt dataset** is publicly available, follow the instructions from the dataset's website. Otherwise, ensure that you have access to the dataset and place it in the correct folder.

### 3. Prepare the Dataset - Image Retrieval
After downloading the dataset, preprocess the images and textual descriptions. Run the following **data_preprocessing.ipynb** to perform the necessary preprocessing:

This script will:
- Clean and preprocess the textual descriptions (tokenization, lemmatization).
- Extract visual features from the artwork images using a pre-trained deep learning model (e.g., ResNet).

### 4. Querying the System
- **Text-Based Query:** Enter a textual description in the search bar to retrieve artworks with similar descriptions.
- **Image-Based Query:** Upload an image to retrieve artworks that are visually similar.

## Evaluation
The system will be evaluated using standard retrieval metrics:
- **Mean Reciprocal Rank (MRR)**
- **Mean Average Precision at 5 (MAP@5)**
- **Recall at 5 (Recall@5)**

## Project Structure

```
semart-multi-modal-retrieval/
│
├── data_preprocessing.ipynb                                        # Script for preprocessing the dataset (images)
├── image_search.ipynb                                              # Script for evaluating retrieval performance
├── artifacts_image_retrieval/image_paths.csv                       # CSV containing the Image Path assignments for the retrieved similar indexes
├── artifacts_image_retrieval/mega_embeddings.pkl                   # Stored embeddings for image retrieval for the entire dataset
├── artifacts_image_retrieval/mega_problematic_images.csv           # CSV containing the corrupted images found in the dataset during preprocessing
├── test_image/                                                     # Folder containing the test images SemArt dataset
└── README.md                                                       # This README file
```

### Notes:
- If you have any questions or issues, feel free to open an issue on the repository's GitHub page.
- Contributions are welcome! Feel free to fork the repository and submit a pull request.

--- 