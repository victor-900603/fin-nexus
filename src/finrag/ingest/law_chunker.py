"""法規條文以「條」為單位切片。"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby

from finrag.ingest.chunker import ChunkConfig
from finrag.models import (
    ParsedRegulation,
    RegulationArticle,
    RegulationChunk,
    RegulationChunkMetadata,
    RegulationMeta,
)
from finrag.text import TokenCounter, pack_by_tokens


def _bracket(parts: Sequence[str]) -> str:
    return "[" + " > ".join(parts) + "]"


def _article_body(article: RegulationArticle) -> str:
    return "\n".join(clause.text for clause in article.clauses)


def _group_articles(
    articles: Sequence[RegulationArticle],
) -> list[tuple[str | None, str | None, list[RegulationArticle]]]:
    groups: list[tuple[str | None, str | None, list[RegulationArticle]]] = []
    for (chapter, section), items in groupby(articles, key=lambda a: (a.chapter, a.section)):
        groups.append((chapter, section, list(items)))
    return groups


def _parent_chunk(
    meta: RegulationMeta,
    chunk_id: str,
    section: str,
    chapter: str | None,
    article_no: str,
    text: str,
) -> RegulationChunk:
    return RegulationChunk(
        text=text,
        metadata=RegulationChunkMetadata(
            doc_id=meta.doc_id,
            chunk_id=chunk_id,
            parent_id=None,
            section=section,
            is_parent=True,
            reg_name=meta.reg_name,
            reg_pcode=meta.reg_pcode,
            article_no=article_no,
            chapter=chapter,
            revision_date=meta.revision_date,
        ),
    )


def _child_chunk(
    meta: RegulationMeta,
    chunk_id: str,
    parent_id: str,
    section: str,
    chapter: str | None,
    article: RegulationArticle,
    text: str,
) -> RegulationChunk:
    return RegulationChunk(
        text=text,
        metadata=RegulationChunkMetadata(
            doc_id=meta.doc_id,
            chunk_id=chunk_id,
            parent_id=parent_id,
            section=section,
            is_parent=False,
            reg_name=meta.reg_name,
            reg_pcode=meta.reg_pcode,
            article_no=article.article_no,
            chapter=chapter,
            revision_date=meta.revision_date,
        ),
    )


def chunk_regulation(
    document: ParsedRegulation,
    counter: TokenCounter,
    config: ChunkConfig | None = None,
) -> list[RegulationChunk]:
    """將法規解析結果切成父子切片：每條一個父切片，條內各款為子切片。"""
    config = config or ChunkConfig()
    meta = document.meta
    doc_id = meta.doc_id
    chunks: list[RegulationChunk] = []
    parent_index = 0
    for chapter, section, articles in _group_articles(document.articles):
        path_parts = [part for part in (chapter, section) if part]
        section_text = " > ".join(path_parts)
        for article in articles:
            body = _article_body(article)
            if not body:
                continue
            prefix = _bracket([meta.reg_name, *path_parts, article.article_no])
            prefix_text = f"{prefix}\n"
            parent_id = f"{doc_id}:p{parent_index}"
            parent_index += 1
            chunks.append(
                _parent_chunk(
                    meta,
                    parent_id,
                    section_text,
                    chapter,
                    article.article_no,
                    prefix_text + body,
                )
            )
            clauses = [(clause.text,) for clause in article.clauses if clause.text]
            groups = pack_by_tokens(
                clauses, lambda item: item[0], counter, config.child_max, counter(prefix_text)
            )
            for child_index, group in enumerate(groups):
                child_text = prefix_text + "\n".join(item[0] for item in group)
                chunks.append(
                    _child_chunk(
                        meta,
                        f"{parent_id}:c{child_index}",
                        parent_id,
                        section_text,
                        chapter,
                        article,
                        child_text,
                    )
                )
    return chunks
