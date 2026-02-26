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

    run_name = os.environ.get("WANDB_RUN_NAME", "").strip()
    if not run_name:
        time_part = datetime.now().strftime("%Y%m%d-%H%M%S")
        ds_part = dataset_name if dataset_name else "dataset"
        run_name = f"{model_name}_{ds_part}_{time_part}"

    config = {
        "model": model_name,
        "dataset": dataset_name,
        "epochs": epochs,
        "output_dir": output_dir,
    }
    if isinstance(extra_config, dict):
        config.update(extra_config)

    wandb_dir = output_dir or "output"
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.login(key=api_key, relogin=False)
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        mode=mode,
        dir=wandb_dir,
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
