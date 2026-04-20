import time
import os

from .models import PaperResult
from .utils import build_semantic_text, normalize_cosine_similarity, normalize_score_map, normalize_space, score_bm25

# sentence-transformers is optional.
# We try to import it, but if it is missing we do not want the whole program
# to crash. Instead, the program can still use BM25-only ranking.
try:
    from sentence_transformers import CrossEncoder, SentenceTransformer, util
except ImportError:
    CrossEncoder = None
    SentenceTransformer = None
    util = None


# Constants are values we treat as fixed.
USER_AGENT = "paper-topic-search/1.0 (Python requests)"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _models_local_only() -> bool:
    """
    Keep local/offline model loading by default, but allow Azure jobs to download
    models when BIOHACKING_MODELS_LOCAL_ONLY=false is set.
    """

    value = os.getenv("BIOHACKING_MODELS_LOCAL_ONLY", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


class HybridRanker:
    """
    This class handles ranking.

    A class is a blueprint.
    HybridRanker is the blueprint for an object that knows how to score papers.

    It combines:
    1. BM25 lexical scoring
    2. semantic embedding similarity
    3. optional cross-encoder reranking
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        bm25_weight: float = 0.45,
        semantic_weight: float = 0.55,
        use_cross_encoder: bool = False,
        cross_encoder_model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        cross_encoder_top_k: int = 25,
    ) -> None:
        """
        This sets up the ranker object.
        """

        self.embedding_model_name = embedding_model_name

        # We normalize the weights so together they add up to 1.
        total_weight = bm25_weight + semantic_weight
        if total_weight <= 0:
            total_weight = 1.0
            bm25_weight = 0.45
            semantic_weight = 0.55

        self.bm25_weight = bm25_weight / total_weight
        self.semantic_weight = semantic_weight / total_weight
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cross_encoder_top_k = cross_encoder_top_k

        # These start empty and are loaded only when needed.
        self._embedding_model = None
        self._cross_encoder = None

        # These booleans tell us whether the optional libraries are available.
        self.semantic_ready = SentenceTransformer is not None and util is not None
        self.cross_encoder_ready = self.use_cross_encoder and CrossEncoder is not None

    def rerank(self, topic: str, results: list[PaperResult]) -> list[PaperResult]:
        """
        Rank the candidate papers from best to worst.
        """

        lexical_ranked = score_bm25(topic, results)
        if not lexical_ranked:
            return []

        if not self.semantic_ready:
            return lexical_ranked

        semantic_ranked = self._score_semantic(topic, lexical_ranked)

        if self.cross_encoder_ready:
            semantic_ranked = self._score_cross_encoder(topic, semantic_ranked)

        return sorted(
            semantic_ranked,
            key=lambda item: (-item.relevance, -item.published.timestamp(), item.title.lower()),
        )

    def _score_semantic(self, topic: str, results: list[PaperResult]) -> list[PaperResult]:
        """
        Compute semantic similarity.

        "Semantic" means meaning-based.
        The idea is that two texts can be related even if they do not share the
        exact same words.
        """

        try:
            model = self._get_embedding_model()
        except Exception:
            return results

        query_text = normalize_space(topic)
        document_texts = [build_semantic_text(paper) for paper in results]

        try:
            query_embedding = model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
            document_embeddings = model.encode(document_texts, convert_to_tensor=True, normalize_embeddings=True)
            similarities = util.cos_sim(query_embedding, document_embeddings)[0]
        except Exception:
            return results

        # BM25 scores and cosine similarity use different numeric scales.
        # We normalize BM25 first so the weighted combination is more sensible.
        normalized_bm25_scores = normalize_score_map([paper.bm25_score for paper in results])

        for index, (paper, similarity) in enumerate(zip(results, similarities)):
            semantic_score = float(similarity)
            paper.semantic_score = semantic_score

            combined = (
                self.bm25_weight * normalized_bm25_scores[index]
                + self.semantic_weight * normalize_cosine_similarity(semantic_score)
            )
            paper.relevance = round(combined, 4)

        return sorted(
            results,
            key=lambda item: (-item.relevance, -item.semantic_score, -item.bm25_score, -item.published.timestamp()),
        )

    def _score_cross_encoder(self, topic: str, results: list[PaperResult]) -> list[PaperResult]:
        """
        Optional final reranking step.

        A cross-encoder is slower but can sometimes judge relevance more precisely.
        Because it is slower, we only run it on the top few results.
        """

        if not results:
            return results

        try:
            model = self._get_cross_encoder()
        except Exception:
            return results

        top_k = min(self.cross_encoder_top_k, len(results))
        finalist_pairs = [(topic, build_semantic_text(paper)) for paper in results[:top_k]]

        try:
            cross_scores = model.predict(finalist_pairs)
        except Exception:
            return results

        max_cross = max(float(score) for score in cross_scores) if len(cross_scores) else 1.0
        min_cross = min(float(score) for score in cross_scores) if len(cross_scores) else 0.0
        span = max(max_cross - min_cross, 1e-9)

        for paper, raw_score in zip(results[:top_k], cross_scores):
            paper.cross_encoder_score = float(raw_score)
            normalized_cross = (paper.cross_encoder_score - min_cross) / span
            paper.relevance = round(0.65 * paper.relevance + 0.35 * normalized_cross, 4)

        return sorted(
            results,
            key=lambda item: (-item.relevance, -item.cross_encoder_score, -item.semantic_score, -item.bm25_score),
        )

    def _get_embedding_model(self):
        """
        Load the embedding model only when needed.
        This is called lazy loading.
        """

        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                local_files_only=_models_local_only(),
            )
        return self._embedding_model

    def _get_cross_encoder(self):
        """
        Load the cross-encoder model only when needed.
        """

        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(
                self.cross_encoder_model_name,
                local_files_only=_models_local_only(),
            )
        return self._cross_encoder
