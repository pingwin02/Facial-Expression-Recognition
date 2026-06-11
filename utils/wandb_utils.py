import os
from datetime import datetime


def _load_dotenv_file(dotenv_path=".env"):
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def _ensure_name_contains_parts(run_name, model_name, dataset_name):
    model_part = str(model_name).strip() if model_name else "model"
    dataset_part = str(dataset_name).strip() if dataset_name else "dataset"

    normalized = str(run_name).strip() if run_name else ""
    lower_name = normalized.lower()

    missing_parts = []
    if model_part.lower() not in lower_name:
        missing_parts.append(model_part)
    if dataset_part.lower() not in lower_name:
        missing_parts.append(dataset_part)

    if not normalized:
        time_part = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{model_part}_{dataset_part}_{time_part}"

    if missing_parts:
        return f"{normalized}_{'_'.join(missing_parts)}"

    return normalized


def init_wandb_run(model_name, dataset_name=None, epochs=None, output_dir=None, extra_config=None):
    _load_dotenv_file(".env")

    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        return None, None

    try:
        import wandb
        from tensorflow.keras.callbacks import Callback
    except Exception:
        print("WANDB_API_KEY found but wandb is not installed. Continuing without wandb logging.")
        return None, None

    project = (
            os.environ.get("WANDB_PROJECT", "facial-expression-recognition").strip() or "facial-expression-recognition"
    )
    entity = os.environ.get("WANDB_ENTITY", "").strip() or None
    mode = os.environ.get("WANDB_MODE", "online").strip() or "online"

    run_name = _ensure_name_contains_parts(
        os.environ.get("WANDB_RUN_NAME", "").strip(),
        model_name=model_name,
        dataset_name=dataset_name,
    )

    config = {
        "model": model_name,
        "dataset": dataset_name,
        "epochs": epochs,
        "output_dir": output_dir,
    }

    try:
        from dataset.loader import CACHE_VERSION

        config["CACHE_VERSION"] = CACHE_VERSION
    except Exception:
        pass

    if isinstance(extra_config, dict):
        config.update(extra_config)

    wandb.login(key=api_key, relogin=False)
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        mode=mode,
        dir="output",
    )

    class _WandbKerasEpochLogger(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            payload = {k: float(v) for k, v in logs.items() if v is not None}
            payload["epoch"] = int(epoch + 1)
            wandb.log(payload, step=int(epoch + 1))

    return run, _WandbKerasEpochLogger()


def finish_wandb_run(run, model_filename=None):
    if run is None:
        return
    try:
        if model_filename:
            run.summary["saved_model_path"] = model_filename
    finally:
        run.finish()
