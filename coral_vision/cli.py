"""Command-line interface for Coral Vision face recognition system."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from coral_vision.config import Paths, get_data_dir_from_env
from coral_vision.core.storage_pgvector import get_storage_backend_from_env
from coral_vision.pipelines.enroll import enroll_person
from coral_vision.pipelines.recognize import recognize_folder

app = typer.Typer(help="Offline face recognition pipeline with pgvector backend.")


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 5000,
    data_dir: Optional[Path] = None,
    use_edgetpu: bool = False,
    debug: bool = False,
    ssl_cert: Optional[Path] = None,
    ssl_key: Optional[Path] = None,
) -> None:
    """Start the Flask web API server.

    For camera access from mobile devices or remote clients, HTTPS is required.
    Generate SSL certificates with: ./generate-local-ssl.sh
    Then use: --ssl-cert certs/cert.pem --ssl-key certs/key.pem
    """
    from coral_vision.web.app import run_server

    if data_dir is None:
        data_dir = get_data_dir_from_env()

    ssl_cert_str = str(ssl_cert) if ssl_cert else None
    ssl_key_str = str(ssl_key) if ssl_key else None

    if ssl_cert_str and ssl_key_str:
        typer.echo(f"🔒 Starting HTTPS server on https://{host}:{port}")
    else:
        typer.echo(f"🌐 Starting HTTP server on http://{host}:{port}")
        if host != "localhost" and host != "127.0.0.1":
            typer.echo(
                "⚠️  Camera access requires HTTPS when accessing via IP address."
            )
            typer.echo("   Generate SSL certs with: ./generate-local-ssl.sh")
            typer.echo("   Then use: --ssl-cert certs/cert.pem --ssl-key certs/key.pem")

    typer.echo(f"Data directory: {data_dir}")
    typer.echo("Storage backend: pgvector")
    typer.echo(f"Edge TPU: {'enabled' if use_edgetpu else 'disabled'}")

    run_server(
        host=host,
        port=port,
        data_dir=data_dir,
        use_edgetpu=use_edgetpu,
        debug=debug,
        ssl_cert=ssl_cert_str,
        ssl_key=ssl_key_str,
    )


@app.command()
def init(
    data_dir: Optional[Path] = None,
) -> None:
    """Initialize database schema and verify connection."""
    if data_dir is None:
        data_dir = get_data_dir_from_env()

    storage = get_storage_backend_from_env()  # noqa: F841
    typer.echo("✅ Database initialized successfully")
    typer.echo(f"Data directory: {data_dir}")


@app.command()
def enroll(
    person_id: str,
    name: str,
    images: Path,
    data_dir: Optional[Path] = None,
    use_edgetpu: bool = False,
    min_score: float = 0.95,
    max_faces: int = 1,
    keep_copies: bool = False,
) -> None:
    """Enroll a person from existing images: detect faces -> crop -> embed -> store in database."""
    if data_dir is None:
        data_dir = get_data_dir_from_env()
    paths = Paths(data_dir=data_dir)

    storage = get_storage_backend_from_env()

    created = enroll_person(
        paths=paths,
        person_id=person_id,
        name=name,
        images_path=images,
        use_edgetpu=use_edgetpu,
        min_score=min_score,
        max_faces=max_faces,
        keep_copies=keep_copies,
        storage=storage,
    )
    typer.echo(json.dumps(created, indent=2))


@app.command()
def recognize(
    input: Path,
    data_dir: Optional[Path] = None,
    use_edgetpu: bool = False,
    threshold: float = 0.6,
    top_k: int = 3,
    per_person_k: int = 20,
    say: bool = False,
    output_json: Optional[Path] = None,
) -> None:
    """Recognize faces in images: detect -> embed -> match -> output results."""
    if data_dir is None:
        data_dir = get_data_dir_from_env()
    paths = Paths(data_dir=data_dir)

    storage = get_storage_backend_from_env()

    results = recognize_folder(
        paths=paths,
        input_path=input,
        use_edgetpu=use_edgetpu,
        threshold=threshold,
        top_k=top_k,
        per_person_k=per_person_k,
        say=say,
        storage=storage,
    )

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        typer.echo(f"Wrote results to: {output_json}")

    typer.echo(json.dumps(results, indent=2))
