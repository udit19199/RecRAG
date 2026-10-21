"""RAPTOR's soft UMAP/GMM summary tree and collapsed-tree cosine retrieval.

Algorithm source: haoyuhan1/RAGvsGraphRAG, commit
d2a0c0c0deb0903d60338d3c416ccd6f9544267c, raptor.py.
We use lists and JSON rather than upstream dictionaries and pickle.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import tiktoken
from pydantic import BaseModel, Field
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from umap import UMAP

from benchmark import RetrievedContext
from cost import indexing
from rag_retrieval import LlamaEmbedding, ResponsesLLM


class TreeNode(BaseModel):
    text: str
    embedding: list[float]
    children: list[int] = Field(default_factory=list)
    layer: int = 0


class Tree(BaseModel):
    nodes: list[TreeNode]


class LayerPlan(BaseModel):
    layer: int
    clusters: list[list[int]]
    summaries: list[str] = Field(default_factory=list)


class TreeProgress(BaseModel):
    tree: Tree
    plans: list[LayerPlan] = Field(default_factory=list)
    fallbacks: list[str] = Field(default_factory=list)


def save_progress(path: Path, progress: TreeProgress) -> None:
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(progress.model_dump_json())
    temporary_path.replace(path)


_cached_tree_path: Path | None = None
_cached_tree: Tree | None = None


def soft_clusters(vectors: np.ndarray) -> list[list[int]]:
    if len(vectors) == 1:
        return [[0]]
    candidates = [
        GaussianMixture(n_components=count, random_state=224, reg_covar=1e-5).fit(
            vectors
        )
        for count in range(1, min(50, len(vectors)))
    ]
    best = min(candidates, key=lambda model: model.bic(vectors))
    fitted = GaussianMixture(
        n_components=best.n_components, random_state=0, reg_covar=1e-5
    ).fit(vectors)
    probabilities = fitted.predict_proba(vectors)
    assignments = [np.flatnonzero(row > 0.1).tolist() for row in probabilities]
    return [
        [index for index, labels in enumerate(assignments) if label in labels]
        for label in range(fitted.n_components)
        if any(label in labels for labels in assignments)
    ]


def cluster_nodes(nodes: list[TreeNode]) -> list[list[int]]:
    if not nodes:
        return []
    if len(nodes) <= 2:
        return [list(range(len(nodes)))]

    vectors = np.asarray([node.embedding for node in nodes])
    global_vectors = UMAP(
        n_neighbors=min(len(nodes) - 1, max(2, int((len(nodes) - 1) ** 0.5))),
        n_components=min(10, len(nodes) - 2),
        metric="cosine",
    ).fit_transform(vectors)
    clusters: list[list[int]] = []
    for group in soft_clusters(global_vectors):
        if len(group) <= 11:
            local_groups = [list(range(len(group)))]
        else:
            local_vectors = UMAP(
                n_neighbors=min(10, len(group) - 1),
                n_components=min(10, len(group) - 2),
                metric="cosine",
            ).fit_transform(vectors[group])
            local_groups = soft_clusters(local_vectors)
        clusters.extend([[group[index] for index in local] for local in local_groups])
    return clusters


def bounded_clusters(
    nodes: list[TreeNode], *, fallbacks: list[str] | None = None
) -> list[list[int]]:
    # 6000 is a soft budget; preserve whole tiny/nonshrinking clusters instead of dropping text.
    tokenizer = tiktoken.get_encoding("cl100k_base")
    groups: list[list[int]] = []
    for group in cluster_nodes(nodes):
        length = sum(len(tokenizer.encode(nodes[index].text)) for index in group)
        if length <= 6000:
            groups.append(group)
            continue
        if len(group) <= 4:
            groups.append(group)
            if fallbacks is not None:
                fallbacks.append(
                    f"tiny oversized group summarized whole: {len(group)} nodes, {length} tokens"
                )
            continue
        nested = cluster_nodes([nodes[index] for index in group])
        if any(len(child_group) >= len(group) for child_group in nested):
            groups.append(group)
            if fallbacks is not None:
                fallbacks.append(
                    f"nonshrinking oversized group: {len(group)} nodes, {length} tokens"
                )
            continue
        for child_group in nested:
            remapped = [group[index] for index in child_group]
            groups.extend(
                [remapped[index] for index in nested_child]
                for nested_child in bounded_clusters(
                    [nodes[index] for index in remapped], fallbacks=fallbacks
                )
            )
    return groups


def build_tree(
    documents: list[str],
    embeddings: LlamaEmbedding,
    llm: ResponsesLLM,
    checkpoint_path: Path | None = None,
) -> Tree:
    if not documents:
        raise ValueError("RAPTOR needs nonempty source documents.")
    # Batch the same vectors to avoid one embedding request per source chunk.
    progress = (
        TreeProgress.model_validate_json(checkpoint_path.read_text())
        if checkpoint_path is not None and checkpoint_path.exists()
        else TreeProgress(tree=Tree(nodes=[]))
    )
    tree = progress.tree
    if tree.nodes and [node.text for node in tree.nodes[: len(documents)]] != documents:
        raise ValueError("RAPTOR checkpoint source documents do not match this corpus.")
    if not tree.nodes:
        tree = Tree(
            nodes=[
                TreeNode(text=text, embedding=embedding)
                for text, embedding in zip(
                    documents,
                    embeddings.get_text_embedding_batch(
                        [text.replace("\n", " ").strip() for text in documents]
                    ),
                    strict=True,
                )
            ]
        )
        progress.tree = tree
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_progress(checkpoint_path, progress)
    current = list(range(len(documents)))
    for layer in range(1, 6):
        if len(current) <= 11:
            break
        parents: list[int] = []
        plan = next((item for item in progress.plans if item.layer == layer), None)
        if plan is None:
            clusters = bounded_clusters(
                [tree.nodes[index] for index in current], fallbacks=progress.fallbacks
            )
            plan = LayerPlan(layer=layer, clusters=clusters)
            progress.plans.append(plan)
            if checkpoint_path is not None:
                save_progress(checkpoint_path, progress)
        for cluster_number, cluster in enumerate(plan.clusters):
            children = [current[index] for index in cluster]
            if any(
                node.layer == layer and node.children == children for node in tree.nodes
            ):
                parents.append(
                    next(
                        i
                        for i, node in enumerate(tree.nodes)
                        if node.layer == layer and node.children == children
                    )
                )
                continue
            if cluster_number < len(plan.summaries):
                summary = plan.summaries[cluster_number]
            else:
                context = "".join(
                    " ".join(tree.nodes[index].text.splitlines()) + "\n\n"
                    for index in children
                )
                generated = llm.infer(
                    [
                        SimpleMessage(
                            role="system", content="You are a helpful assistant."
                        ).model_dump(),
                        SimpleMessage(
                            role="user",
                            content=(
                                "Write a summary of the following, including as many key details as possible. "
                                f"Use at most 100 tokens:\n{context}"
                            ),
                        ).model_dump(),
                    ],
                    max_output_tokens=8192,
                )
                if not generated or not generated[0].strip():
                    raise RuntimeError("RAPTOR summary generation returned empty text.")
                summary = generated[0]
                tokens = tiktoken.get_encoding("cl100k_base").encode(summary)
                if len(tokens) > 100:
                    # Approximation: cap persisted text; generation-time max_tokens differs.
                    progress.fallbacks.append(
                        f"summary capped from {len(tokens)} to 100 tokens"
                    )
                    summary = tiktoken.get_encoding("cl100k_base").decode(tokens[:100])
                plan.summaries.append(summary)
                if checkpoint_path is not None:
                    save_progress(checkpoint_path, progress)
            embedding = embeddings.get_text_embedding(
                summary.replace("\n", " ").strip()
            )
            parents.append(len(tree.nodes))
            tree.nodes.append(
                TreeNode(
                    text=summary, embedding=embedding, children=children, layer=layer
                )
            )
            if checkpoint_path is not None:
                save_progress(checkpoint_path, progress)
        current = parents
        if checkpoint_path is not None:
            save_progress(checkpoint_path, progress)
    return tree


class SimpleMessage(BaseModel):
    role: str
    content: str


def retrieve(
    *,
    documents: list[str],
    question: str,
    index_dir: Path,
    embeddings: LlamaEmbedding,
    llm: ResponsesLLM,
    top_k: int,
) -> RetrievedContext:
    global _cached_tree_path, _cached_tree

    started = perf_counter()
    tree_path = index_dir / "raptor_tree.json"
    with indexing():
        if tree_path.exists():
            if _cached_tree_path == tree_path and _cached_tree is not None:
                tree = _cached_tree
            else:
                tree = Tree.model_validate_json(tree_path.read_text())
                _cached_tree_path = tree_path
                _cached_tree = tree
            construction_seconds = 0.0
        else:
            tree = build_tree(
                documents, embeddings, llm, tree_path.with_name("raptor_progress.json")
            )
            temporary_path = tree_path.with_suffix(".tmp")
            temporary_path.write_text(tree.model_dump_json())
            temporary_path.replace(tree_path)
            _cached_tree_path = tree_path
            _cached_tree = tree
            construction_seconds = perf_counter() - started
    started = perf_counter()
    query = np.asarray(
        embeddings.get_query_embedding(question.replace("\n", " ").strip())
    )
    vectors = np.asarray([node.embedding for node in tree.nodes])
    distances = cdist([query], vectors, metric="cosine")[0]
    tokenizer = tiktoken.get_encoding("cl100k_base")
    selected: list[int] = []
    tokens = 0
    for index in np.argsort(distances)[:top_k].tolist():
        length = len(tokenizer.encode(tree.nodes[index].text))
        if tokens + length > 3500:
            break
        selected.append(index)
        tokens += length
    return RetrievedContext(
        passages=[tree.nodes[index].text for index in selected],
        scores=[float(1 - distances[index]) for index in selected],
        construction_seconds=construction_seconds,
        retrieval_seconds=perf_counter() - started,
    )


if __name__ == "__main__":
    from benchmark import main

    main(default_method="raptor")
