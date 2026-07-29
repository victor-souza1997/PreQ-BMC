from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Relatorio nao encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value: Any) -> str:
    if value is None:
        return "n/d"
    return f"{100.0 * float(value):.2f}%"


def _seconds(value: Any) -> str:
    if value is None:
        return "n/d"
    return f"{float(value):.3f} s"


def _yes_no(value: Any) -> str:
    return "sim" if bool(value) else "nao"


def _selected_result(experiment: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    refined = experiment.get("quality_refined", {})
    if refined.get("accepted") is True:
        return "quality_refined", refined
    return "formal_only", experiment.get("formal_only", {})


def render_summary(run_dir: Path) -> str:
    reports_dir = run_dir / "reports"
    experiment = _load_json(reports_dir / "experiment_summary.json")
    pipeline = _load_json(reports_dir / "pipeline_summary.json")
    method, result = _selected_result(experiment)

    benchmark = experiment.get("benchmark", {})
    reference = experiment.get("reference", {})
    deployment = result.get("deployment_metrics", {})
    resources = result.get("resource_metrics", {})
    timing = pipeline.get("timing_metrics", {})
    controls = pipeline.get("resource_controls", {})
    status_counts = pipeline.get("esbmc_status_counts", {})
    chaining = pipeline.get("chaining_ok", {})

    q_values = result.get("Q", [])
    i_values = result.get("I", [])
    f_values = result.get("F", [])
    formats = [
        f"  camada {index}: <Q={q}, I={i}, F={f}>"
        for index, (q, i, f) in enumerate(zip(q_values, i_values, f_values, strict=False))
    ]

    lines = [
        "=== RESUMO DA EXECUCAO PREQ-BMC ===",
        f"Diretorio: {run_dir}",
        "",
        "REGIAO LOCAL",
        f"  dataset: {benchmark.get('dataset', 'n/d')}",
        f"  arquitetura solicitada: {benchmark.get('arch', 'n/d')}",
        f"  amostra: {benchmark.get('sample_id', 'n/d')}",
        f"  epsilon L_infinito: {benchmark.get('eps', 'n/d')}",
        f"  classe prevista / classe real: {reference.get('predicted_label', 'n/d')} / {reference.get('sample_label', 'n/d')}",
        f"  margem limpa: {reference.get('clean_margin', 'n/d')}",
        "",
        "RESULTADO SELECIONADO",
        f"  secao do relatorio: {method}",
        f"  aceito: {_yes_no(result.get('accepted', result.get('success', False)))}",
        f"  status final: {result.get('final_status', 'UNKNOWN')}",
        f"  nivel de garantia: {result.get('guarantee_level', 'unknown')}",
        f"  contrato: {result.get('contract_status', 'UNKNOWN')}",
        f"  nao saturacao formal: {result.get('no_saturation_status', 'SKIPPED')}",
        f"  equivalencia inteira Python/C: {_yes_no(result.get('python_c_exact_match'))}",
        "",
        "FORMATOS FIXOS POR CAMADA",
        *(formats or ["  nenhum formato selecionado"]),
        "",
        "QUALIDADE DE IMPLANTACAO",
        f"  amostras comparadas: {benchmark.get('samples_evaluated', 'n/d')}",
        f"  acuracia Keras quantizado: {_percent(deployment.get('quantized_keras_accuracy'))}",
        f"  acuracia Python fixo: {_percent(deployment.get('python_fixed_accuracy'))}",
        f"  acuracia C fixo: {_percent(deployment.get('c_fixed_accuracy'))}",
        f"  divergencia de classes vs. Keras: {_percent(deployment.get('mismatch_rate_vs_keras'))}",
        f"  maior erro absoluto de logit: {deployment.get('max_abs_logit_error', 'n/d')}",
        f"  maior taxa de saturacao: {_percent(deployment.get('max_saturation_rate'))}",
        f"  compressao de parametros vs. FP32: {resources.get('compression_ratio_vs_float32', 'n/d')}x",
        "",
        "RECURSOS",
        f"  solver MILP: {controls.get('solver', 'n/d')}",
        f"  perfil ESBMC: {controls.get('esbmc_profile', 'n/d')}",
        f"  chamadas ESBMC: {timing.get('number_of_esbmc_calls', status_counts.get('esbmc_total_count', 'n/d'))}",
        f"  chamadas verificadas: {status_counts.get('esbmc_verified_count', 'n/d')}",
        f"  candidatos rejeitados pelo ESBMC: {status_counts.get('esbmc_failed_count', 'n/d')}",
        f"  timeouts / memouts: {status_counts.get('esbmc_timeout_count', 'n/d')} / {status_counts.get('esbmc_memout_count', 'n/d')}",
        f"  tempo total: {_seconds(timing.get('total_runtime_seconds'))}",
        f"  tempo total no ESBMC: {_seconds(timing.get('total_esbmc_time_seconds'))}",
        "",
        "LIMITE DA AFIRMACAO",
        f"  soundness registrada: {pipeline.get('soundness', 'unknown')}",
        f"  encadeamento entre contratos: {_yes_no(chaining.get('all_ok'))}",
    ]
    if result.get("guarantee_level") == "harness-verified":
        lines.extend(
            [
                "  Interpretacao: os harnesses e as checagens de implantacao passaram,",
                "  mas esta execucao nao estabelece a garantia deployed-transfer.",
            ]
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mostra um resumo em portugues de uma execucao PreQ-BMC.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("output/video_demo"),
        help="Diretorio da execucao que contem reports/.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        print(render_summary(args.run_dir))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Erro: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
