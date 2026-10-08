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
    vectors = np.asarray([node.embedding for node in nodes])
    global_vectors = UMAP(
        n_neighbors=int((len(nodes) - 1) ** 0.5),
        n_components=min(10, len(nodes) - 2),
        metric="cosine",
    ).fit_transform(vectors)
    clusters: list[list[int]] = []
    for group in soft_clusters(global_vectors):
        if len(group) <= 11:
            clusters.append(group)
            continue
        local_vectors = UMAP(
            n_neighbors=10,
            n_components=10,
            metric="cosine",
        ).fit_transform(vectors[group])
        clusters.extend(
            [
                [group[index] for index in local]
                for local in soft_clusters(local_vectors)
            ]
        )
    return clusters


def bounded_clusters(
    nodes: list[TreeNode], *, parent_size: int | None = None
) -> list[list[int]]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    groups: list[list[int]] = []
    for group in cluster_nodes(nodes):
        length = sum(len(tokenizer.encode(nodes[index].text)) for index in group)
        if length <= 6000 or len(group) == 1:
            groups.append(group)
            continue
        if parent_size is not None and len(group) >= parent_size:
            # Fail rather than fabricate clusters when upstream recursion stalls.
            raise RuntimeError(
                "RAPTOR reclustering did not reduce an oversized cluster. "
                "No source-order split was applied."
            )
        groups.extend(
            [group[index] for index in nested]
            for nested in bounded_clusters(
                [nodes[index] for index in group], parent_size=len(group)
            )
        )
    return groups


def build_tree(
    documents: list[str], embeddings: LlamaEmbedding, llm: ResponsesLLM
) -> Tree:
    if not documents:
        raise ValueError("RAPTOR needs nonempty source documents.")
    tree = Tree(
        nodes=[
            TreeNode(text=text, embedding=embeddings.get_text_embedding(text))
            for text in documents
        ]
    )
    current = list(range(len(tree.nodes)))
    for layer in range(1, 6):
        if len(current) <= 11:
            break
        parents: list[int] = []
        for cluster in bounded_clusters([tree.nodes[index] for index in current]):
            children = [current[index] for index in cluster]
            context = "\n\n".join(
                " ".join(tree.nodes[index].text.splitlines()) for index in children
            )
            summary = llm.infer(
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
                ]
            )[0]
            parents.append(len(tree.nodes))
            tree.nodes.append(
                TreeNode(
                    text=summary,
                    embedding=embeddings.get_text_embedding(summary),
                    children=children,
                    layer=layer,
                )
            )
        current = parents
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
    started = perf_counter()
    tree_path = index_dir / "raptor_tree.json"
    with indexing():
        if tree_path.exists():
            tree = Tree.model_validate_json(tree_path.read_text())
            construction_seconds = 0.0
        else:
            tree = build_tree(documents, embeddings, llm)
            tree_path.write_text(tree.model_dump_json())
            construction_seconds = perf_counter() - started
    started = perf_counter()
    query = np.asarray(embeddings.get_query_embedding(question))
    vectors = np.asarray([node.embedding for node in tree.nodes])
    scores = (vectors @ query) / np.maximum(
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query), 1e-12
    )
    tokenizer = tiktoken.get_encoding("cl100k_base")
    selected: list[int] = []
    tokens = 0
    for index in np.argsort(-scores, kind="stable")[:top_k].tolist():
        length = len(tokenizer.encode(tree.nodes[index].text))
        if tokens + length > 3500:
            break
        selected.append(index)
        tokens += length
    return RetrievedContext(
        passages=[tree.nodes[index].text for index in selected],
        scores=[float(scores[index]) for index in selected],
        construction_seconds=construction_seconds,
        retrieval_seconds=perf_counter() - started,
    )


if __name__ == "__main__":
    from benchmark import main

    main(default_method="raptor")
