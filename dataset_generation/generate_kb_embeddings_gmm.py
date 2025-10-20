import argparse
import asyncio
import json
import math
import os
import numpy as np
import psutil
import umap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from numpy.lib.format import open_memmap
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from atlaskv.gpt_session import GPT
from atlaskv.gpt_session_async import GPTAsync
from atlaskv.utils.data_utils import DataPoint

# Environment setup
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "text-embedding-3-large"
    dataset_name: str = "synthetic_data_qkv"
    endpoint_url: Optional[str] = None
    endpoint_api_key: Optional[str] = None
    dataset_path: Optional[str] = None
    output_path: str = "your_dataset_dir"
    max_concurrency: int = 10
    batch_size: int = 8192
    random: bool = False
    generating_embeddings: bool = False
    chunked_processing: bool = False
    chunk_size: int = 100000
    max_chunks: Optional[int] = None
    cluster: bool = False
    gmm_n_init: int = 2
    gmm_max_iter: int = 50
    n_layers: int = 2
    umap_dim: int = 64
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"


class MemoryMonitor:
    """Monitors system memory usage."""
    
    @staticmethod
    def check_memory() -> bool:
        """Check if memory usage is within acceptable limits."""
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            print(f"\033[91mWARNING: CPU MEM USAGE {memory.percent}%. SHUTTING DOWN...\033[0m")
            return False
        return True


class EmbeddingProcessor:
    """Handles embedding computation for different models."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.monitor = MemoryMonitor()

    def compute_sentence_transformer_embeddings(
        self, texts: List[str], model_name: str, batch_size: int
    ) -> np.ndarray:
        """Compute embeddings using SentenceTransformer."""
        model = SentenceTransformer(model_name, device="cuda", cache_folder="your_cache_dir")
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            if not self.monitor.check_memory():
                break
            batch = texts[i:i + batch_size]
            embd = model.encode(batch, convert_to_numpy=True)
            embeddings.append(embd)
        
        return np.concatenate(embeddings, axis=0)

    async def compute_async_embeddings(
        self, texts: List[str], model_name: str, endpoint_url: str, endpoint_api_key: str
    ) -> List[List[float]]:
        """Compute embeddings using async GPT."""
        gpt = GPTAsync(model_name, endpoint_url, endpoint_api_key)
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        
        async def get_embedding_with_semaphore(text: str):
            async with semaphore:
                return await gpt.generate_embedding(text)
        
        tasks = [get_embedding_with_semaphore(text) for text in texts]
        embeddings = []
        
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating embeddings"):
            embedding = await task
            embeddings.append(embedding)
        
        return embeddings

    def compute_sync_embeddings(
        self, texts: List[str], model_name: str, endpoint_url: str, endpoint_api_key: str
    ) -> List[List[float]]:
        """Compute embeddings using sync GPT."""
        gpt = GPT(model_name, endpoint_url, endpoint_api_key)
        batch_size = self.config.batch_size
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = gpt.generate_embeddings_batch(batch) or []
            embeddings.extend(batch_embeddings)
        
        return embeddings


class StreamingProcessor:
    """Handles streaming embedding generation for large datasets."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.monitor = MemoryMonitor()

    def process_chunked_embeddings(self, dataset_path: str, save_name: str) -> int:
        """Process large datasets in chunks."""
        print(f"\033[94m[Starting streaming chunked processing for large dataset]\033[0m")
        
        if self.config.model_name == "all-MiniLM-L6-v2":
            model = SentenceTransformer(self.config.model_name, device="cuda:0", cache_folder="your_cache_dir")
        else:
            raise NotImplementedError(f"Chunked processing not implemented for model {self.config.model_name}")
        
        chunk_files = []
        total_processed = 0
        chunk_idx = 0
        
        chunk_dir = f"{self.config.output_path}/chunks_{self.config.dataset_name}_{save_name}"
        os.makedirs(chunk_dir, exist_ok=True)
        
        with open(dataset_path, 'r') as f:
            chunk_data = []
            
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    chunk_data.append(DataPoint(**data))
                except Exception as e:
                    print(f"Error loading line {line_num}: {e}")
                    continue
                
                if len(chunk_data) >= self.config.chunk_size:
                    print(f"\033[94m[Processing chunk {chunk_idx + 1} with {len(chunk_data)} records]\033[0m")
                    
                    if not self.monitor.check_memory():
                        print("Insufficient memory, stopping...")
                        break
                    
                    chunk_key_file, chunk_val_file = self._process_single_chunk(
                        chunk_data, chunk_idx, chunk_dir, model
                    )
                    
                    if chunk_key_file and chunk_val_file:
                        chunk_files.append((chunk_key_file, chunk_val_file))
                        total_processed += len(chunk_data)
                        print(f"Chunk {chunk_idx + 1} completed, total processed: {total_processed}")
                    
                    del chunk_data
                    chunk_data = []
                    chunk_idx += 1
                    
                    if self.config.max_chunks and chunk_idx >= self.config.max_chunks:
                        print(f"Reached maximum chunks limit: {self.config.max_chunks}")
                        break
            
            # Process remaining data
            if chunk_data:
                print(f"\033[94m[Processing final chunk {chunk_idx + 1} with {len(chunk_data)} records]\033[0m")
                
                if self.monitor.check_memory():
                    chunk_key_file, chunk_val_file = self._process_single_chunk(
                        chunk_data, chunk_idx, chunk_dir, model
                    )
                    
                    if chunk_key_file and chunk_val_file:
                        chunk_files.append((chunk_key_file, chunk_val_file))
                        total_processed += len(chunk_data)
                        print(f"Final chunk completed, total processed: {total_processed}")
                
                del chunk_data
        
        print(f"\033[94m[All chunks processed, merging {len(chunk_files)} chunk files]\033[0m")
        self._merge_chunk_files(chunk_files, save_name)
        self._cleanup_chunks(chunk_files, chunk_dir)
        
        print(f"\033[92m[Streaming processing completed! Total {total_processed} records processed]\033[0m")
        return total_processed

    def _process_single_chunk(self, chunk_data: List[DataPoint], chunk_idx: int, chunk_dir: str, model) -> Tuple[Optional[str], Optional[str]]:
        """Process a single chunk of data."""
        try:
            key_texts = [data.key_string for data in chunk_data]
            value_texts = [data.description for data in chunk_data]
            
            # Process key embeddings
            key_embeddings = []
            for i in tqdm(range(0, len(key_texts), self.config.batch_size)):
                batch = key_texts[i:i + self.config.batch_size]
                embd = model.encode(batch, convert_to_numpy=True)
                key_embeddings.append(embd)
            
            # Process value embeddings
            val_embeddings = []
            for i in tqdm(range(0, len(value_texts), self.config.batch_size)):
                if not self.monitor.check_memory():
                    break
                batch = value_texts[i:i + self.config.batch_size]
                embd = model.encode(batch, convert_to_numpy=True)
                val_embeddings.append(embd)
            
            # Concatenate and save
            key_embeddings = np.concatenate(key_embeddings, axis=0)
            val_embeddings = np.concatenate(val_embeddings, axis=0)
            
            chunk_key_file = f"{chunk_dir}/chunk_{chunk_idx:06d}_key.npy"
            chunk_val_file = f"{chunk_dir}/chunk_{chunk_idx:06d}_val.npy"
            
            np.save(chunk_key_file, key_embeddings.astype(np.float32))
            np.save(chunk_val_file, val_embeddings.astype(np.float32))
            
            del key_embeddings, val_embeddings, key_texts, value_texts
            return chunk_key_file, chunk_val_file
            
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {e}")
            return None, None

    def _merge_chunk_files(self, chunk_files: List[Tuple[str, str]], save_name: str):
        """Merge all chunk files into final embedding files."""
        if not chunk_files:
            print("No chunk files to merge")
            return
        
        print(f"Merging {len(chunk_files)} chunk files...")
        
        # Load first chunk to get dimensions
        first_key_file, first_val_file = chunk_files[0]
        first_key_embd = np.load(first_key_file)
        first_val_embd = np.load(first_val_file)
        
        key_dim = first_key_embd.shape[1]
        val_dim = first_val_embd.shape[1]
        total_size = sum(np.load(key_file).shape[0] for key_file, _ in chunk_files)
        
        # Create final memory-mapped files
        final_key_file = f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key.npy"
        final_val_file = f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_value.npy"
        
        key_mm = open_memmap(final_key_file, mode='w+', dtype=np.float32, shape=(total_size, key_dim))
        val_mm = open_memmap(final_val_file, mode='w+', dtype=np.float32, shape=(total_size, val_dim))
        
        # Merge chunks
        current_idx = 0
        for chunk_key_file, chunk_val_file in tqdm(chunk_files, desc="Merging chunks"):
            key_embd = np.load(chunk_key_file)
            val_embd = np.load(chunk_val_file)
            
            chunk_size = key_embd.shape[0]
            end_idx = current_idx + chunk_size
            
            key_mm[current_idx:end_idx] = key_embd.astype(np.float32)
            val_mm[current_idx:end_idx] = val_embd.astype(np.float32)
            
            current_idx = end_idx
            del key_embd, val_embd
        
        del key_mm, val_mm
        print(f"Merge completed! Final files saved to {final_key_file} and {final_val_file}")

    def _cleanup_chunks(self, chunk_files: List[Tuple[str, str]], chunk_dir: str):
        """Clean up temporary chunk files."""
        for chunk_key_file, chunk_val_file in chunk_files:
            try:
                os.remove(chunk_key_file)
                os.remove(chunk_val_file)
            except:
                pass
        
        try:
            os.rmdir(chunk_dir)
        except:
            pass


class ConsistencyChecker:
    """Checks consistency of generated embeddings."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(a @ b / denom)

    def _window_spot_check(
        self, 
        emb_matrix: np.ndarray, 
        texts: List[str], 
        recompute_batch_fn, 
        indices: List[int], 
        window: int = 3, 
        tol: float = 1e-3
    ) -> List[Tuple[int, int, float, float]]:
        """Sample embeddings and check if they are correct."""
        problems = []
        batch_size = max(1, min(32, len(indices)))
        
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            recomputed = recompute_batch_fn(batch_texts)
            
            for local_k, i in enumerate(batch_idx):
                e = recomputed[local_k]
                win_start = max(0, i - window)
                win_end = min(len(texts), i + window + 1)
                sims = [self._compute_similarity(e, emb_matrix[j]) for j in range(win_start, win_end)]
                j_star = win_start + int(np.argmax(sims))
                sim_i = sims[int(i - win_start)]
                sim_star = max(sims)
                
                if j_star != i and (sim_i + tol) < sim_star:
                    problems.append((int(i), int(j_star), float(sim_i), float(sim_star)))
        
        return problems

    def check_consistency(
        self, 
        key_embeds: np.ndarray, 
        value_embeds: np.ndarray, 
        dataset: List[DataPoint]
    ):
        """Check consistency of generated embeddings."""
        try:
            key_texts = [data.key_string for data in dataset]
            value_texts = [data.description for data in dataset]

            rng = np.random.default_rng(2025)
            num_samples = int(min(200, len(key_texts)))
            sample_indices = sorted(rng.choice(len(key_texts), size=num_samples, replace=False).tolist()) if len(key_texts) > 0 else []

            if len(sample_indices) > 0:
                if self.config.model_name == "all-MiniLM-L6-v2":
                    st_model = SentenceTransformer(self.config.model_name, device="cuda", cache_folder="your_cache_dir")
                    
                    def recompute_keys(text_batch: List[str]):
                        return st_model.encode(text_batch, convert_to_numpy=True)
                    
                    def recompute_vals(text_batch: List[str]):
                        return st_model.encode(text_batch, convert_to_numpy=True)
                else:
                    gpt_checker = GPT(self.config.model_name, self.config.endpoint_url, self.config.endpoint_api_key)
                    
                    def recompute_keys(text_batch: List[str]):
                        return np.asarray(gpt_checker.generate_embeddings_batch(text_batch) or [], dtype=np.float32)
                    
                    def recompute_vals(text_batch: List[str]):
                        return np.asarray(gpt_checker.generate_embeddings_batch(text_batch) or [], dtype=np.float32)

                key_problems = self._window_spot_check(key_embeds, key_texts, recompute_keys, sample_indices, window=3, tol=1e-3)
                val_problems = self._window_spot_check(value_embeds, value_texts, recompute_vals, sample_indices, window=3, tol=1e-3)

                print("\n===== Consistency Check Report =====")
                print(f"Checked samples: {len(sample_indices)} (window=3)")
                print(f"Key mismatches: {len(key_problems)}")
                if len(key_problems) > 0:
                    print(f"Key examples (up to 5): {key_problems[:5]}")
                print(f"Value mismatches: {len(val_problems)}")
                if len(val_problems) > 0:
                    print(f"Value examples (up to 5): {val_problems[:5]}")
                print("===================================\n")
            else:
                print("[Consistency Check] Skipped (dataset not available in this run context).")
        except Exception as e:
            print(f"[Consistency Check] Failed with error: {e}")


class HierarchicalClusterer:
    """Handles hierarchical clustering of embeddings."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def perform_clustering(self, key_embeds: np.ndarray, save_name: str):
        """Perform hierarchical clustering on embeddings."""
        keys = np.array(key_embeds, dtype=np.float32)
        print(f"\033[94m[UMAP dimension reduction from {keys.shape[1]} to {self.config.umap_dim} dimensions]\033[0m")
        
        reducer = umap.UMAP(
            n_components=self.config.umap_dim,
            n_neighbors=self.config.umap_n_neighbors,
            min_dist=self.config.umap_min_dist,
            metric=self.config.umap_metric,
            random_state=42,
            verbose=True
        )
        keys_low = reducer.fit_transform(keys.astype(np.float32))

        for i in range(self.config.n_layers):
            if i == 0:
                keys_last = keys_low
                original_keys_last = keys
            
            k = int(math.ceil(math.pow(len(keys_low), (self.config.n_layers - i) / (self.config.n_layers + 1))))
            print(f"\033[94m[Hierarchical Clustering {len(keys_last)} keys with {i+1} / {self.config.n_layers} layers and {k} clusters]\033[0m")
            
            gmm = GaussianMixture(
                n_components=k,
                n_init=self.config.gmm_n_init,
                max_iter=self.config.gmm_max_iter,
                random_state=42,
                verbose=1,
                init_params='k-means++'
            ).fit(keys_last)
            
            labels = gmm.predict(keys_last)
            centers = gmm.means_
            
            # Create mappings
            clust2idlist_mapping = {int(c): [int(i) for i in range(len(keys_last)) if labels[i] == c] for c in range(k)}
            id2clust_mapping = {int(i): int(c) for i, c in enumerate(labels)}
            
            # Compute cluster means with original keys
            original_centers = np.mean(original_keys_last[clust2idlist_mapping[0]], axis=0)
            for c in range(1, k):
                original_centers = np.concatenate([original_centers, np.mean(original_keys_last[clust2idlist_mapping[c]], axis=0)])
            original_centers = original_centers.reshape(-1, original_keys_last.shape[-1])

            # Save results
            if i == self.config.n_layers - 1:
                np.save(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key_root.npy", original_centers.astype(np.float32))
                with open(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key_root_c2id_mapping.json", "w") as f:
                    json.dump(clust2idlist_mapping, f)
                with open(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key_root_id2c_mapping.json", "w") as f:
                    json.dump(id2clust_mapping, f)
            else:
                np.save(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key_inter{self.config.n_layers - 1 - i}.npy", original_centers.astype(np.float32))
                with open(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key_inter{self.config.n_layers - 1 - i}_c2id_mapping.json", "w") as f:
                    json.dump(clust2idlist_mapping, f)
                with open(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key_inter{self.config.n_layers - 1 - i}_id2c_mapping.json", "w") as f:
                    json.dump(id2clust_mapping, f)
            
            keys_last = centers
            original_keys_last = original_centers


class EmbeddingGenerator:
    """Main orchestrator for embedding generation."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.processor = EmbeddingProcessor(config)
        self.streaming_processor = StreamingProcessor(config)
        self.consistency_checker = ConsistencyChecker(config)
        self.clusterer = HierarchicalClusterer(config)

    def _get_save_name(self) -> str:
        """Get save name based on model."""
        if self.config.model_name == "all-MiniLM-L6-v2":
            return "all-MiniLM-L6-v2"
        elif self.config.model_name in ["ada-embeddings", "text-embedding-ada-002"]:
            return "OAI"
        else:
            return "BigOAI"

    def _load_dataset(self) -> List[DataPoint]:
        """Load dataset from file."""
        if not any(split in self.config.dataset_name for split in ["train", "val", "test"]):
            with open(self.config.dataset_path, "r") as file:
                loaded_dataset = json.loads(file.read())
                return [DataPoint(**line) for line in loaded_dataset]
        return []

    async def _generate_embeddings_streaming(self, dataset: List[DataPoint], save_name: str):
        """Generate embeddings with streaming write."""
        if self.config.model_name == "all-MiniLM-L6-v2":
            await self._generate_sentence_transformer_streaming(dataset, save_name)
        elif self.config.model_name in ["ada-embeddings", "text-embedding-3-large", "text-embedding-ada-002"]:
            await self._generate_gpt_streaming(dataset, save_name)
        else:
            raise ValueError(f"Model {self.config.model_name} not supported.")

    async def _generate_sentence_transformer_streaming(self, dataset: List[DataPoint], save_name: str):
        """Generate embeddings using SentenceTransformer with streaming."""
        key_texts = [data.key_string for data in dataset]
        value_texts = [data.description for data in dataset]

        model = SentenceTransformer(self.config.model_name, device="cuda", cache_folder="your_cache_dir")
        batch_size = self.config.batch_size

        # Key embeddings streaming
        total_n = len(key_texts)
        first_batch = key_texts[:batch_size]
        first_embd = model.encode(first_batch, convert_to_numpy=True)
        key_dim = int(first_embd.shape[1])
        key_file = f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key.npy"
        key_mm = open_memmap(key_file, mode='w+', dtype=np.float32, shape=(total_n, key_dim))
        key_mm[0:len(first_batch)] = first_embd.astype(np.float32)

        for i in tqdm(range(batch_size, total_n, batch_size), desc="Generating key_string embeddings"):
            if not MemoryMonitor.check_memory():
                break
            batch_texts = key_texts[i:i + batch_size]
            embd = model.encode(batch_texts, convert_to_numpy=True)
            key_mm[i:i+len(batch_texts)] = embd.astype(np.float32)
        del key_mm

        # Value embeddings streaming
        total_n_v = len(value_texts)
        first_batch_v = value_texts[:batch_size]
        first_embd_v = model.encode(first_batch_v, convert_to_numpy=True)
        val_dim = int(first_embd_v.shape[1])
        val_file = f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_value.npy"
        val_mm = open_memmap(val_file, mode='w+', dtype=np.float32, shape=(total_n_v, val_dim))
        val_mm[0:len(first_batch_v)] = first_embd_v.astype(np.float32)

        for i in tqdm(range(batch_size, total_n_v, batch_size), desc="Generating description embeddings"):
            if not MemoryMonitor.check_memory():
                break
            batch_texts = value_texts[i:i + batch_size]
            embd = model.encode(batch_texts, convert_to_numpy=True)
            val_mm[i:i+len(batch_texts)] = embd.astype(np.float32)
        del val_mm

    async def _generate_gpt_streaming(self, dataset: List[DataPoint], save_name: str):
        """Generate embeddings using GPT with streaming."""
        if not self.config.random:
            await self._generate_gpt_sync_streaming(dataset, save_name)
        else:
            await self._generate_gpt_async_streaming(dataset, save_name)

    async def _generate_gpt_sync_streaming(self, dataset: List[DataPoint], save_name: str):
        """Generate embeddings using sync GPT with streaming."""
        gpt = GPT(self.config.model_name, self.config.endpoint_url, self.config.endpoint_api_key)
        batch_size = self.config.batch_size

        # Keys
        print("Generating key_string embeddings...")
        key_texts = [data.key_string for data in dataset]
        total_n = len(key_texts)
        key_file = f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key.npy"
        
        first_batch = key_texts[:batch_size]
        first_embeddings = gpt.generate_embeddings_batch(first_batch) or []
        if len(first_embeddings) == 0:
            raise RuntimeError("Failed to generate first batch of key embeddings")
        
        key_dim = int(len(first_embeddings[0]))
        key_mm = open_memmap(key_file, mode='w+', dtype=np.float32, shape=(total_n, key_dim))
        key_mm[0:len(first_embeddings)] = np.asarray(first_embeddings, dtype=np.float32)
        
        for i in tqdm(range(batch_size, total_n, batch_size), desc="Generating key_string embeddings"):
            batch_texts = key_texts[i:i + batch_size]
            batch_embeddings = gpt.generate_embeddings_batch(batch_texts) or []
            if len(batch_embeddings) != len(batch_texts):
                print(f"Warning: Key_string embedding generation failed for batch {i//batch_size + 1}")
            if len(batch_embeddings) > 0:
                key_mm[i:i+len(batch_embeddings)] = np.asarray(batch_embeddings, dtype=np.float32)
        del key_mm

        # Values
        print("Generating description embeddings...")
        value_texts = [data.description for data in dataset]
        total_nv = len(value_texts)
        val_file = f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_value.npy"
        
        first_batch_v = value_texts[:batch_size]
        first_embeddings_v = gpt.generate_embeddings_batch(first_batch_v) or []
        if len(first_embeddings_v) == 0:
            raise RuntimeError("Failed to generate first batch of value embeddings")
        
        val_dim = int(len(first_embeddings_v[0]))
        val_mm = open_memmap(val_file, mode='w+', dtype=np.float32, shape=(total_nv, val_dim))
        val_mm[0:len(first_embeddings_v)] = np.asarray(first_embeddings_v, dtype=np.float32)
        
        for i in tqdm(range(batch_size, total_nv, batch_size), desc="Generating description embeddings"):
            batch_texts = value_texts[i:i + batch_size]
            batch_embeddings = gpt.generate_embeddings_batch(batch_texts) or []
            if len(batch_embeddings) != len(batch_texts):
                print(f"Warning: Description embedding generation failed for batch {i//batch_size + 1}")
            if len(batch_embeddings) > 0:
                val_mm[i:i+len(batch_embeddings)] = np.asarray(batch_embeddings, dtype=np.float32)
        del val_mm

    async def _generate_gpt_async_streaming(self, dataset: List[DataPoint], save_name: str):
        """Generate embeddings using async GPT with streaming."""
        gpt = GPTAsync(self.config.model_name, self.config.endpoint_url, self.config.endpoint_api_key)
        
        async def stream_async(part: str, outfile: str):
            all_elements = [getattr(data, part) for data in dataset]
            semaphore = asyncio.Semaphore(self.config.max_concurrency)

            async def get_one(idx: int, text: str):
                async with semaphore:
                    emb = await gpt.generate_embedding(text)
                    return idx, emb

            # Fetch first to get dimension
            first_idx, first_text = 0, all_elements[0]
            _, first_emb = await get_one(first_idx, first_text)
            dim = int(len(first_emb))
            mm = open_memmap(outfile, mode='w+', dtype=np.float32, shape=(len(all_elements), dim))
            mm[first_idx] = np.asarray(first_emb, dtype=np.float32)

            tasks = [asyncio.create_task(get_one(i, t)) for i, t in enumerate(all_elements) if i != first_idx]
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Generating {part} embeddings"):
                idx, emb = await coro
                mm[idx] = np.asarray(emb, dtype=np.float32)
            del mm

        await stream_async("key_string", f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key.npy")
        await stream_async("description", f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_value.npy")

    async def generate_embeddings(self):
        """Main method to generate embeddings."""
        os.makedirs(self.config.output_path, exist_ok=True)
        save_name = self._get_save_name()

        if self.config.chunked_processing:
            print(f"\033[94m[Using chunked processing mode]\033[0m")
            if self.config.generating_embeddings:
                self.streaming_processor.process_chunked_embeddings(self.config.dataset_path, save_name)
            key_embeds = np.load(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key.npy")
            value_embeds = np.load(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_value.npy")
        else:
            dataset = self._load_dataset()
            
            if self.config.generating_embeddings:
                await self._generate_embeddings_streaming(dataset, save_name)
                key_embeds = np.load(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key.npy")
                value_embeds = np.load(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_value.npy")
            else:
                key_embeds = np.load(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_key.npy")
                value_embeds = np.load(f"{self.config.output_path}/{self.config.dataset_name}_{save_name}_embd_value.npy")

        # Check consistency
        if 'dataset' in locals():
            self.consistency_checker.check_consistency(key_embeds, value_embeds, dataset)

        # Perform clustering if requested
        if self.config.cluster:
            self.clusterer.perform_clustering(key_embeds, save_name)


def create_config_from_args(args: argparse.Namespace) -> EmbeddingConfig:
    """Create configuration from command line arguments."""
    return EmbeddingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        endpoint_url=args.endpoint_url,
        endpoint_api_key=args.endpoint_api_key,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_concurrency=args.max_concurrency,
        batch_size=args.batch_size,
        random=args.random,
        generating_embeddings=args.generating_embeddings,
        chunked_processing=args.chunked_processing,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        cluster=args.cluster,
        gmm_n_init=args.gmm_n_init,
        gmm_max_iter=args.gmm_max_iter,
        n_layers=args.n_layers,
        umap_dim=args.umap_dim,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="text-embedding-3-large",
        choices=["all-MiniLM-L6-v2", "text-embedding-3-large", "ada-embeddings", "text-embedding-ada-002"],
    )
    parser.add_argument("--dataset_name", type=str, default="synthetic_data_QA_augmented")
    parser.add_argument("--endpoint_url", type=str)
    parser.add_argument("--endpoint_api_key", type=str)
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to the dataset in JSON format.")
    parser.add_argument("--output_path", type=str, default="your_dataset_dir")
    parser.add_argument("--max_concurrency", type=int, default=10, help="Maximum number of concurrent API calls")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for embedding generation")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--generating_embeddings", action="store_true")
    parser.add_argument("--chunked_processing", action="store_true", help="Enable chunked processing for large datasets")
    parser.add_argument("--chunk_size", type=int, default=100000, help="Size of each chunk for processing")
    parser.add_argument("--max_chunks", type=int, default=None, help="Maximum number of chunks to process (None for all)")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--gmm_n_init", type=int, default=2, help="Number of initializations for GMM")
    parser.add_argument("--gmm_max_iter", type=int, default=50, help="Maximum number of iterations for GMM")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers for hierarchical clustering")
    parser.add_argument("--umap_dim", type=int, default=64, help="Number of dimensions for UMAP")
    parser.add_argument("--umap_n_neighbors", type=int, default=30, help="Number of neighbors for UMAP")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="Minimum distance for UMAP")
    parser.add_argument("--umap_metric", type=str, default="cosine", help="Metric for UMAP")
    return parser


async def main():
    """Main function."""
    args = build_parser().parse_args()
    config = create_config_from_args(args)
    
    generator = EmbeddingGenerator(config)
    await generator.generate_embeddings()


if __name__ == "__main__":
    asyncio.run(main())