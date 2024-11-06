import os
import streamlit as st
import pymupdf4llm
from pathlib import Path
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from llama_index.core import Document, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

# Set the image directory
image_path = Path("images")  # Define the path for the images
image_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = ""

# Streamlit app title
st.title("PDF Content Extraction and Retrieval")

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  # Store the path of the temp file

    # Use the temporary file path for processing
    docs = pymupdf4llm.to_markdown(doc=temp_file_path,  # Use the temporary file path
                                    page_chunks=True,
                                    write_images=True,
                                    image_path=str(image_path),  # Use the newly created directory
                                    image_format="jpg")
    llama_documents = []

    for document in docs:
        metadata = {
            "file_path": document["metadata"].get("file_path"),
            "page": str(document["metadata"].get("page")),
            "images": str(document.get("images")),
            "toc_items": str(document.get("toc_items")),
        }

        llama_document = Document(
            text=document["text"],
            metadata=metadata,
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )

        llama_documents.append(llama_document)

    # Initialize Qdrant Client
    client = qdrant_client.QdrantClient(location=":memory:")

    # Create collections for text and image data
    client.create_collection(
        collection_name="text_collection",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    client.create_collection(
        collection_name="image_collection",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )

    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )
    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection"
    )

    # Create MultiModal index
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )
    image_documents = SimpleDirectoryReader(str(image_path)).load_data()

    index = MultiModalVectorStoreIndex.from_documents(
        llama_documents + image_documents,
        storage_context=storage_context)

    # Set query input
    query = st.text_input("Enter your query here:")
    if query:
        retriever = index.as_retriever(similarity_top_k=1, image_similarity_top_k=1)
        retrieval_results = retriever.retrieve(query)

        # Display retrieval results
        st.subheader("Retrieval Results:")
        for res_node in retrieval_results:
            if isinstance(res_node.node, ImageNode):
                st.image(res_node.node.metadata["file_path"], caption="Retrieved Image")
            else:
                st.write("Text Node:", res_node.node.text)

        # Function to plot images
        def plot_images(image_paths):
            images_shown = 0
            fig, axarr = plt.subplots(2, 3, figsize=(16, 9))
            for img_path in image_paths:
                if os.path.isfile(img_path):
                    image = Image.open(img_path)
                    axarr[images_shown // 3, images_shown % 3].imshow(image)
                    axarr[images_shown // 3, images_shown % 3].axis('off')
                    images_shown += 1
                    if images_shown >= 6:
                        break
            plt.tight_layout()
            st.pyplot(fig)

        retrieved_image = []
        for res_node in retrieval_results:
            if isinstance(res_node.node, ImageNode):
                retrieved_image.append(res_node.node.metadata["file_path"])

        plot_images(retrieved_image)
