#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evalúa Recall@K de predicciones M9.2 contra archivos_codigo del dataset."""
import argparse
import re
from pathlib import Path
import pandas as pd


def parse_files(value):
    """Convierte una celda separada por ';' en rutas normalizadas."""
    if pd.isna(value):
        return set()
    return {
        re.sub(r"/+", "/", item.strip().replace("\\", "/")).lstrip("./")
        for item in str(value).split(";")
        if item.strip()
    }


def main():
    parser = argparse.ArgumentParser(description="Compara Recall@K de BM25, búsqueda semántica y Weighted RRF.")
    parser.add_argument("--dataset", required=True, help="CSV original con archivos_codigo.")
    parser.add_argument("--predictions", required=True,
                        help="predicciones_comparativas.csv (o predicciones_m92.csv para evaluar solo RRF).")
    parser.add_argument("--out", default="evaluacion_recall")
    parser.add_argument("--partition", default=None,
                        help="Valor de particion a evaluar (por ejemplo, prueba). Si se omite, usa todas las filas.")
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    predictions_path = Path(args.predictions)
    if not dataset_path.is_file() or not predictions_path.is_file():
        raise SystemExit("No se encuentra el dataset o el archivo de predicciones indicado.")
    if any(k < 1 for k in args.ks):
        raise SystemExit("Todos los valores de K deben ser positivos.")

    data = pd.read_csv(dataset_path).fillna("")
    preds = pd.read_csv(predictions_path).fillna("")
    for col in ["id", "archivos_codigo"]:
        if col not in data.columns:
            raise SystemExit(f"El dataset no contiene la columna '{col}'.")
    for col in ["ticket", "archivo", "rank"]:
        if col not in preds.columns:
            raise SystemExit(f"Las predicciones no contienen la columna '{col}'.")

    if args.partition is not None:
        if "particion" not in data.columns:
            raise SystemExit("Se indicó --partition, pero el dataset no contiene 'particion'.")
        data = data[data["particion"].astype(str) == args.partition]
    if data.empty:
        raise SystemExit("No hay tickets en la partición seleccionada.")

    preds["ticket"] = preds["ticket"].astype(str)
    preds["rank"] = pd.to_numeric(preds["rank"], errors="coerce")
    if "metodo" not in preds.columns:
        preds["metodo"] = "Weighted_RRF"
    methods = [m for m in ["BM25", "Semantico", "Weighted_RRF"] if m in set(preds["metodo"].astype(str))]
    if not methods:
        raise SystemExit("No se encontraron rankings BM25, Semantico o Weighted_RRF.")

    rows = []
    summary = []
    for method in methods:
        method_preds = preds[preds["metodo"].astype(str) == method]
        by_ticket = {ticket: group.sort_values("rank") for ticket, group in method_preds.groupby("ticket")}
        method_rows = []
        total_relevant = 0
        totals_found = {k: 0 for k in args.ks}

        for _, ticket in data.iterrows():
            tid = str(ticket["id"])
            relevant = parse_files(ticket["archivos_codigo"])
            if not relevant:
                method_rows.append({"metodo": method, "ticket": tid, "n_archivos_reales": 0,
                                    "estado": "sin_ground_truth"})
                continue
            group = by_ticket.get(tid)
            ranked = group["archivo"].astype(str).tolist() if group is not None else []
            row = {"metodo": method, "ticket": tid, "n_archivos_reales": len(relevant),
                   "estado": "evaluado" if ranked else "sin_predicciones"}
            total_relevant += len(relevant)
            for k in args.ks:
                top_k = set(parse_files(";".join(ranked[:k])))
                found = relevant.intersection(top_k)
                row[f"encontrados@{k}"] = len(found)
                row[f"Recall@{k}"] = len(found) / len(relevant)
                row[f"archivos_encontrados@{k}"] = ";".join(sorted(found))
                totals_found[k] += len(found)
            method_rows.append(row)

        evaluated = pd.DataFrame(method_rows)
        valid = evaluated[evaluated["n_archivos_reales"] > 0].copy()
        if valid.empty:
            continue
        rows.extend(method_rows)
        for k in args.ks:
            summary.append({"metodo": method, "metrica": f"Recall@{k}",
                            "macro_por_ticket": valid[f"Recall@{k}"].mean(),
                            "micro_por_archivo": totals_found[k] / total_relevant if total_relevant else 0.0,
                            "tickets_evaluados": len(valid),
                            "archivos_relevantes_totales": total_relevant,
                            "archivos_recuperados_totales": totals_found[k]})

    detail = pd.DataFrame(rows)
    if detail.empty or not summary:
        raise SystemExit("No hay tickets con archivos_codigo válidos para evaluar.")

    out_base = Path(args.out)
    detail_path = out_base.with_name(out_base.name + "_detalle.csv")
    summary_path = out_base.with_name(out_base.name + "_resumen.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nRESULTADOS COMPARATIVOS DE RECALL")
    for item in summary:
        print(f"{item['metodo']} | {item['metrica']}: macro={item['macro_por_ticket']:.4f} | micro={item['micro_por_archivo']:.4f}")
    print(f"\nDetalle: {detail_path}")
    print(f"Resumen: {summary_path}")
    if args.partition is None and "particion" in data.columns:
        print("AVISO: no se filtró ninguna partición; para el resultado experimental, especifica --partition con el conjunto de prueba.")


if __name__ == "__main__":
    main()
