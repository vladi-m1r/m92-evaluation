#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M9.2 reconstruido con recuperación del snapshot anterior a cada ticket."""
import argparse
import ast
import io
import json
import re
import subprocess
from collections import OrderedDict
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_CHARS = 1500
INITIAL_CHUNKS = 6
GUIDED_CHUNKS = 12
RRF_K = 60
BM25_WEIGHT = 0.80
SEMANTIC_WEIGHT = 0.20


def tokenize(text):
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^\W\d_]+|\d+", str(text).lower(), flags=re.UNICODE)


def split_python(path, root):
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    rel = path.relative_to(root).as_posix()
    lines = source.splitlines()
    chunks = []
    try:
        tree = ast.parse(source)
        nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        nodes.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "end_lineno", 0)))
        for node in nodes:
            start = max(1, getattr(node, "lineno", 1))
            end = min(len(lines), getattr(node, "end_lineno", start))
            body = "\n".join(lines[start - 1:end])[:MAX_CHARS]
            if body.strip():
                chunks.append({"file": rel, "text": f"FILE: {rel}\nSYMBOL: {getattr(node, 'name', '')}\n{body}"})
    except SyntaxError:
        pass
    if not chunks:
        for i in range(0, len(lines), 35):
            body = "\n".join(lines[i:i + 35])[:MAX_CHARS]
            if body.strip():
                chunks.append({"file": rel, "text": f"FILE: {rel}\n{body}"})
    return chunks


def load_repo(repo):
    files = sorted(p for p in repo.rglob("*.py") if ".git" not in p.parts and "__pycache__" not in p.parts)
    file_chunks = {}
    for path in files:
        rel = path.relative_to(repo).as_posix()
        chunks = split_python(path, repo)
        if chunks:
            file_chunks[rel] = chunks
    return file_chunks, len(files)


def query_text(row):
    title = str(row.get("summary", row.get("title", "")) or "")
    body = str(row.get("texto_modelo", "") or "")
    if not body.strip() or body.lower() == "nan":
        body = str(row.get("description", row.get("descripcion_sin_enlaces", "")) or "")
    return (title + "\n" + body).strip()


class EmbeddingCache:
    """Caché LRU de embeddings por texto para reutilizar código entre tickets/snapshots."""
    def __init__(self, model, max_items=30000, batch_size=64):
        self.model = model
        self.max_items = max(0, int(max_items))
        self.batch_size = int(batch_size)
        self.vectors = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_many(self, texts):
        unique_missing, seen_missing = [], set()
        for text in texts:
            if text not in self.vectors and text not in seen_missing:
                unique_missing.append(text)
                seen_missing.add(text)
        newly_encoded = {}
        if unique_missing:
            encoded = self.model.encode(
                unique_missing, normalize_embeddings=True,
                show_progress_bar=False, batch_size=self.batch_size,
                convert_to_numpy=True,
            )
            for text, vector in zip(unique_missing, encoded):
                vector = np.asarray(vector, dtype=np.float32)
                newly_encoded[text] = vector
                self.misses += 1
                if self.max_items > 0:
                    self.vectors[text] = vector
                    self.vectors.move_to_end(text)
                    while len(self.vectors) > self.max_items:
                        self.vectors.popitem(last=False)
        result = []
        for text in texts:
            if text in newly_encoded:
                # Usar el vector recién calculado incluso si el LRU lo desalojó.
                result.append(newly_encoded[text])
            else:
                vector = self.vectors[text]
                self.hits += 1
                self.vectors.move_to_end(text)
                result.append(vector)
        return np.vstack(result) if result else np.empty((0, 0), dtype=np.float32)


class SnapshotIndex:
    """Precalcula la tokenización y BM25 local una sola vez por snapshot."""
    def __init__(self, file_chunks):
        self.file_chunks = file_chunks
        self.bm25_by_file = {}
        self.flat_chunks = []
        self.flat_tokens = []
        self.file_offsets = []
        offset = 0
        for filename, chunks in file_chunks.items():
            tokens = [tokenize(chunk["text"]) for chunk in chunks]
            self.bm25_by_file[filename] = BM25Okapi(tokens) if tokens else None
            self.file_offsets.append((filename, offset, len(chunks)))
            self.flat_chunks.extend(chunks)
            self.flat_tokens.extend(tokens)
            offset += len(chunks)

    def select_chunks(self, query):
        query_tokens = tokenize(query)
        selected_indices, seen = [], set()
        for filename, offset, count in self.file_offsets:
            bm_local = self.bm25_by_file[filename]
            guided_idx = []
            if bm_local is not None and count:
                scores = bm_local.get_scores(query_tokens)
                guided_idx = sorted(range(count), key=lambda i: scores[i], reverse=True)[:GUIDED_CHUNKS]
            for local_idx in list(range(min(INITIAL_CHUNKS, count))) + guided_idx:
                global_idx = offset + local_idx
                if global_idx not in seen:
                    selected_indices.append(global_idx)
                    seen.add(global_idx)
        return selected_indices


def retrieve_one(row, index, model, embedding_cache, topn=20):
    """Devuelve rankings independientes de BM25, semántico y Weighted RRF."""
    query = query_text(row)
    if not query:
        raise ValueError(f"Ticket {row.get('id', '?')} no tiene summary/description.")
    selected_indices = index.select_chunks(query)
    if not selected_indices:
        return {"BM25": [], "Semantico": [], "Weighted_RRF": []}
    selected = [index.flat_chunks[i] for i in selected_indices]
    selected_tokens = [index.flat_tokens[i] for i in selected_indices]

    bm = BM25Okapi(selected_tokens)
    chunk_scores = bm.get_scores(tokenize(query))
    bm_file = {}
    for chunk, score in zip(selected, chunk_scores):
        bm_file[chunk["file"]] = max(bm_file.get(chunk["file"], float("-inf")), float(score))
    bm_rank = sorted(bm_file, key=bm_file.get, reverse=True)

    texts = [chunk["text"] for chunk in selected]
    embeddings = embedding_cache.get_many(texts)
    qemb = model.encode([query], normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True)[0]
    sims = embeddings @ np.asarray(qemb, dtype=np.float32)
    sem_file = {}
    for chunk, score in zip(selected, sims):
        sem_file[chunk["file"]] = max(sem_file.get(chunk["file"], float("-inf")), float(score))
    sem_rank = sorted(sem_file, key=sem_file.get, reverse=True)

    fused = {}
    for rank, filename in enumerate(bm_rank, 1):
        fused[filename] = fused.get(filename, 0.0) + BM25_WEIGHT / (RRF_K + rank)
    for rank, filename in enumerate(sem_rank, 1):
        fused[filename] = fused.get(filename, 0.0) + SEMANTIC_WEIGHT / (RRF_K + rank)
    rrf_rank = sorted(fused, key=fused.get, reverse=True)

    rankings = {"BM25": bm_rank, "Semantico": sem_rank, "Weighted_RRF": rrf_rank}
    scores = {"BM25": bm_file, "Semantico": sem_file, "Weighted_RRF": fused}
    return {
        method: [{"rank": i + 1, "archivo": filename, "score": scores[method][filename]}
                 for i, filename in enumerate(ranking[:topn])]
        for method, ranking in rankings.items()
    }


def git_root(repo):
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"La ruta no es un repositorio Git válido: {repo}\n{result.stderr.strip()}")
    return Path(result.stdout.strip()).resolve()


def materialize_commit(repo, commit, destination):
    """Extrae el árbol de un commit sin cambiar HEAD ni el working tree del usuario."""
    check = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{tree}}"],
                           capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(f"No se encuentra el commit/árbol {commit}. Comprueba que el clon tenga ese historial.")
    archive = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit],
                             capture_output=True)
    if archive.returncode:
        raise RuntimeError(f"No se pudo extraer el commit {commit}: {archive.stderr.decode(errors='replace')}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
        # El repositorio fuente debe ser confiable (por ejemplo, el clon oficial de Django).
        tar.extractall(path=destination)


def main():
    parser = argparse.ArgumentParser(description="M9.2 con snapshot histórico por ticket (commit padre).")
    parser.add_argument("--repo", required=True, help="Clon Git local del repositorio Django.")
    parser.add_argument("--input", required=True, help="CSV con id, summary, texto_modelo y padre.")
    parser.add_argument("--out", default="resultados_m92_historico")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--ticket", type=str, default=None)
    parser.add_argument("--embedding-cache-size", type=int, default=30000,
                        help="Máximo de fragmentos en caché (0 desactiva la caché).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Tamaño de lote para generar embeddings (en GPU se puede aumentar si hay VRAM).")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Dispositivo para embeddings: auto selecciona CUDA si está disponible.")
    parser.add_argument("--snapshot", choices=["historical", "current"], default="historical",
                        help="historical usa padre por ticket; current usa el estado actual del clon.")
    args = parser.parse_args()

    repo = git_root(Path(args.repo).resolve())
    csv_path = Path(args.input).resolve()
    out = Path(args.out).resolve()
    if not csv_path.is_file():
        raise SystemExit(f"No existe el CSV: {csv_path}")
    if args.top < 1:
        raise SystemExit("--top debe ser mayor que cero.")
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).fillna("")
    required = {"id", "summary"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Faltan columnas obligatorias: {', '.join(sorted(missing))}")
    if args.snapshot == "historical" and "padre" not in df.columns:
        raise SystemExit("La evaluación histórica requiere la columna 'padre'.")
    if args.ticket is not None:
        df = df[df["id"].astype(str) == str(args.ticket)]
    if df.empty:
        raise SystemExit("No hay tickets para procesar.")

    if args.embedding_cache_size < 0 or args.batch_size < 1:
        raise SystemExit("--embedding-cache-size debe ser >= 0 y --batch-size debe ser >= 1.")
    # Selección explícita del dispositivo para acelerar la generación de embeddings.
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("No se pudo importar PyTorch. Instala una versión compatible con tu entorno.") from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Solicitaste --device cuda, pero PyTorch no detecta una GPU CUDA. Verifica drivers y la instalación de PyTorch.")

    print(f"Cargando modelo de embeddings: {MODEL_NAME} en {device.upper()}...", flush=True)
    if device == "cuda":
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Ejecutando sin gpu")
    model = SentenceTransformer(MODEL_NAME, device=device)
    embedding_cache = EmbeddingCache(model, args.embedding_cache_size, args.batch_size)
    rows = []
    comparative_rows = []
    metadata_snapshots = {}

    if args.snapshot == "current":
        print("Indexando el estado actual del repositorio (no histórico)...", flush=True)
        file_chunks, nfiles = load_repo(repo)
        print(f"Archivos Python: {nfiles}; archivos indexados: {len(file_chunks)}", flush=True)
        index = SnapshotIndex(file_chunks)
        print(f"Fragmentos preparados: {len(index.flat_chunks)}", flush=True)
        for _, row in df.iterrows():
            print(f"Procesando ticket {row['id']}...", flush=True)
            preds = retrieve_one(row, index, model, embedding_cache, args.top)
            tid = row["id"]
            for method, ranking in preds.items():
                for pred in ranking:
                    comparative_rows.append({"ticket": tid, "snapshot_commit": "CURRENT_WORKTREE", "metodo": method, **pred})
                    if method == "Weighted_RRF":
                        rows.append({"ticket": tid, "snapshot_commit": "CURRENT_WORKTREE",
                                     "score_rrf": pred["score"], "score_bm25": None,
                                     "score_semantico": None, "rank": pred["rank"], "archivo": pred["archivo"]})
            print(f"Ticket {tid}: {len(preds)} candidatos")
        metadata_snapshots["CURRENT_WORKTREE"] = {"python_files_detected": nfiles, "files_indexed": len(file_chunks)}
    else:
        # Agrupar por commit evita volver a indexar el mismo snapshot para cada ticket.
        df["_parent"] = df["padre"].astype(str).str.strip()
        missing_parent = df[df["_parent"].isin(["", "nan", "None"])]["id"].tolist()
        if missing_parent:
            raise SystemExit(f"Tickets sin commit padre en 'padre': {missing_parent[:20]}")
        for commit, group in df.groupby("_parent", sort=False):
            print(f"\nPreparando snapshot {commit} ({len(group)} ticket(s))...", flush=True)
            with tempfile.TemporaryDirectory(prefix="m92_snapshot_") as tmp:
                snapshot_dir = Path(tmp)
                print("  Extrayendo commit y analizando archivos Python...", flush=True)
                materialize_commit(repo, commit, snapshot_dir)
                file_chunks, nfiles = load_repo(snapshot_dir)
                nchunks = sum(len(v) for v in file_chunks.values())
                index = SnapshotIndex(file_chunks)
                metadata_snapshots[commit] = {"tickets": int(len(group)), "python_files_detected": nfiles,
                                              "files_indexed": len(file_chunks), "chunks_indexed": nchunks}
                print(f"Archivos Python: {nfiles}; archivos indexados: {len(file_chunks)}; fragmentos: {nchunks}", flush=True)
                for _, row in group.iterrows():
                    print(f"  Procesando ticket {row['id']}...", flush=True)
                    preds = retrieve_one(row, index, model, embedding_cache, args.top)
                    tid = row["id"]
                    for method, ranking in preds.items():
                        for pred in ranking:
                            comparative_rows.append({"ticket": tid, "snapshot_commit": commit, "metodo": method, **pred})
                            if method == "Weighted_RRF":
                                rows.append({"ticket": tid, "snapshot_commit": commit,
                                             "rank": pred["rank"], "archivo": pred["archivo"],
                                             "score_rrf": pred["score"], "score_bm25": None,
                                             "score_semantico": None})
                    rrf_top5 = [p["archivo"] for p in preds["Weighted_RRF"][:5]]
                    print(f"Ticket {tid}: Top-5 RRF = " + ", ".join(rrf_top5))

    result = pd.DataFrame(rows, columns=["ticket", "snapshot_commit", "rank", "archivo", "score_rrf", "score_bm25", "score_semantico"])
    result.to_csv(out / "predicciones_m92.csv", index=False, encoding="utf-8-sig")
    comparative = pd.DataFrame(comparative_rows,
        columns=["ticket", "snapshot_commit", "metodo", "rank", "archivo", "score"])
    comparative.to_csv(out / "predicciones_comparativas.csv", index=False, encoding="utf-8-sig")
    meta = {"method": "M9.2 reconstruido", "snapshot_mode": args.snapshot,
            "snapshot_rule": "commit padre (padre) por ticket" if args.snapshot == "historical" else "working tree actual",
            "embedding_model": MODEL_NAME, "bm25_weight": BM25_WEIGHT,
            "semantic_weight": SEMANTIC_WEIGHT, "rrf_k": RRF_K,
            "initial_chunks_per_file": INITIAL_CHUNKS, "guided_chunks_per_file": GUIDED_CHUNKS,
            "max_chars_per_chunk": MAX_CHARS, "repo": str(repo),
            "embedding_cache_max_items": args.embedding_cache_size,
            "embedding_batch_size": args.batch_size, "embedding_device": device,
            "embedding_cache_hits": embedding_cache.hits,
            "embedding_cache_misses": embedding_cache.misses,
            "tickets_processed": int(len(df)), "snapshots": metadata_snapshots}
    (out / "configuracion_m92.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPredicciones guardadas en: {out / 'predicciones_m92.csv'}")
    print(f"Rankings comparativos guardados en: {out / 'predicciones_comparativas.csv'}")
    print(f"Configuración guardada en: {out / 'configuracion_m92.json'}")
    print(f"Dispositivo de embeddings: {device.upper()}")
    print(f"Caché de embeddings: {embedding_cache.hits} reutilizados; "
          f"{embedding_cache.misses} calculados; {len(embedding_cache.vectors)} en memoria.")


if __name__ == "__main__":
    main()
