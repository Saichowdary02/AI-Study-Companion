import chromadb
import os
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Create the directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        
        # Check for API key before initializing embeddings
        def get_openai_api_key():
            """Get OpenAI API key from environment variables"""
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key.startswith("#") or len(api_key.strip()) == 0:
                return None
            return api_key.strip()

        api_key = get_openai_api_key()
        if not api_key:
            print("Warning: No valid OpenAI API key found for embeddings. Using text-based search.")
        else:
            print(f"✅ OpenAI API key loaded for embeddings (length: {len(api_key)})")

        try:
            if not api_key:
                raise ValueError("OpenAI API key not set. Please configure your API key in the .env file.")
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        except Exception as e:
            self.embeddings = None
            print(f"Warning: Failed to initialize OpenAI embeddings: {e}")
            print("Vector store will use text-based search instead of semantic search.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased for fewer chunks
            chunk_overlap=100  # Reduced for faster processing
        )

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Handle collection creation with proper embedding dimensions
        if self.embeddings is None:
            # Use default embedding function when OpenAI is not available
            try:
                self.collection = self.client.get_or_create_collection(
                    name="study_notes",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"Error creating collection without embeddings: {e}")
                # Try to recreate the collection
                try:
                    self.client.delete_collection("study_notes")
                except:
                    pass
                self.collection = self.client.create_collection(
                    name="study_notes",
                    metadata={"hnsw:space": "cosine"}
                )
        else:
            # Handle embedding dimension mismatch
            try:
                self.collection = self.client.get_collection("study_notes")
                print("Found existing collection, checking embedding compatibility...")
                # Test if we can add a small embedding to check dimensions
                try:
                    test_embedding = self.embeddings.embed_query("test")
                    # Try to add a test document to see if dimensions match
                    self.collection.add(
                        documents=["test"],
                        embeddings=[test_embedding],
                        ids=["test_dimension_check"]
                    )
                    # If successful, remove the test document
                    self.collection.delete(ids=["test_dimension_check"])
                    print("✅ Collection embedding dimensions are compatible")
                except Exception as dim_error:
                    if "dimension" in str(dim_error).lower():
                        print(f"⚠️  Embedding dimension mismatch detected: {dim_error}")
                        print("🔄 Recreating collection with correct embedding dimensions...")
                        # Delete the old collection
                        self.client.delete_collection("study_notes")
                        # Create new collection
                        self.collection = self.client.create_collection(
                            name="study_notes",
                            metadata={"hnsw:space": "cosine"}
                        )
                        print("✅ Created new collection with text-embedding-3-small (1536 dimensions)")
                    else:
                        raise dim_error
            except Exception:
                # Collection doesn't exist, create new one
                self.collection = self.client.create_collection(
                    name="study_notes",
                    metadata={"hnsw:space": "cosine"}
                )
                print("✅ Created new collection with text-embedding-3-small")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None) -> None:
        """Add documents to the vector store"""
        if metadatas is None:
            metadatas = [{}] * len(documents)

        # Split documents into chunks
        all_chunks = []
        all_metadatas = []
        chunk_ids = []

        for doc_idx, (doc, metadata) in enumerate(zip(documents, metadatas)):
            chunks = self.text_splitter.split_text(doc)
            # Limit chunks per document to prevent excessive processing
            max_chunks_per_doc = 20  # Limit to 20 chunks per document
            chunks = chunks[:max_chunks_per_doc]

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    **metadata,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'document_index': doc_idx
                })
                chunk_ids.append(f"doc_{doc_idx}_chunk_{chunk_idx}")

        try:
            if self.embeddings is None:
                # Fallback: store documents without embeddings
                print("Warning: OpenAI API key not set. Storing documents without embeddings.")
                # Add to ChromaDB without embeddings (will use default embedding function)
                self.collection.add(
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=chunk_ids
                )
            else:
                # Generate embeddings and add to collection
                embeddings = self.embeddings.embed_documents(all_chunks)
                # Add to ChromaDB with embeddings
                self.collection.add(
                    documents=all_chunks,
                    embeddings=embeddings,
                    metadatas=all_metadatas,
                    ids=chunk_ids
                )

            print(f"Successfully added {len(all_chunks)} document chunks to vector store")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, similarity_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Search for similar documents with high similarity filtering"""
        if self.embeddings is None:
            # Fallback: use text-based search without embeddings
            print("Warning: OpenAI API key not set. Using basic text search.")
            try:
                # Get all documents and search through them
                all_docs = self.collection.get(limit=self.collection.count())
                if not all_docs or not all_docs.get('documents'):
                    return []

                # Simple text matching with relevance scoring
                query_lower = query.lower()
                matches = []
                for i, doc in enumerate(all_docs['documents']):
                    doc_lower = doc.lower()
                    # Calculate simple relevance score based on keyword matches
                    query_words = set(query_lower.split())
                    doc_words = set(doc_lower.split())
                    common_words = query_words.intersection(doc_words)
                    relevance_score = len(common_words) / len(query_words) if query_words else 0

                    if relevance_score > 0.3:  # At least 30% keyword match
                        matches.append({
                            'content': doc,
                            'metadata': all_docs['metadatas'][i] if all_docs.get('metadatas') else {},
                            'distance': 1 - relevance_score,  # Convert relevance to distance (lower is better)
                            'relevance_score': relevance_score
                        })

                # Sort by relevance (lowest distance first)
                matches.sort(key=lambda x: x['distance'])

                # Filter by similarity threshold
                filtered_matches = [m for m in matches if m['relevance_score'] >= similarity_threshold]

                # If no matches meet threshold, return top matches anyway (but mark as low confidence)
                if not filtered_matches and matches:
                    filtered_matches = matches[:k]
                    for match in filtered_matches:
                        match['low_confidence'] = True

                return filtered_matches[:k]
            except Exception as e:
                print(f"Error in fallback search: {e}")
                return []

        try:
            query_embedding = self.embeddings.embed_query(query)

            # Get optimized number of results for faster processing
            n_results = max(int(k * 1.5), 8)  # Ensure it's always an integer
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Format results and filter by similarity
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 1.0
                    # Convert distance to similarity score (cosine similarity)
                    similarity_score = 1 - distance

                    if similarity_score >= similarity_threshold:
                        formatted_results.append({
                            'content': doc,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                            'distance': distance,
                            'similarity_score': similarity_score
                        })

            # Sort by similarity score (highest first)
            formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)

            # If no results meet threshold, return top results with low confidence flag
            if not formatted_results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0][:k]):
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 1.0
                    similarity_score = 1 - distance
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': distance,
                        'similarity_score': similarity_score,
                        'low_confidence': True
                    })

            return formatted_results[:k]
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name
        }
        
    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        try:
            # Get all document IDs
            if self.collection.count() > 0:
                # Get all document IDs in batches to handle large collections
                all_ids = []
                offset = 0
                batch_size = 1000

                while True:
                    results = self.collection.get(limit=batch_size, offset=offset)
                    if not results or not results.get('ids'):
                        break

                    batch_ids = results['ids']
                    all_ids.extend(batch_ids)

                    if len(batch_ids) < batch_size:
                        break
                    offset += batch_size

                # Delete all documents from the collection in batches
                if all_ids:
                    for i in range(0, len(all_ids), batch_size):
                        batch = all_ids[i:i + batch_size]
                        self.collection.delete(ids=batch)
                        print(f"Deleted batch of {len(batch)} documents")

            print(f"Collection cleared, remaining documents: {self.collection.count()}")

        except Exception as e:
            print(f"Error clearing collection: {e}")
            # As a fallback, try to recreate the collection
            try:
                self.client.delete_collection("study_notes")
                self.collection = self.client.create_collection(
                    name="study_notes",
                    metadata={"hnsw:space": "cosine"}
                )
                print("Collection recreated as fallback")
            except Exception as recreate_error:
                print(f"Error recreating collection: {recreate_error}")
                raise

# Global vector store instance
vector_store = VectorStore(persist_directory="./chroma_db")
