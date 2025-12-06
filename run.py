import argparse
import subprocess
import time
from datetime import datetime

BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def log_step(step, text):
    # Log d'entrée d'une étape du pipeline avec horodatage
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{BLUE}[{now}] {step} — {text}{RESET}")


def log_ok(duration):
    # Log de fin d'étape avec durée d'exécution
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{GREEN}[{now}] terminé ({duration:.2f}s){RESET}\n")


def run_cmd(cmd, capture=False):
    # Exécution robuste de commandes système (make, python, gcloud, etc.)
    start = time.time()
    try:
        if capture:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
            log_ok(time.time() - start)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"{RED}Erreur: {e}{RESET}")
        exit(1)

    log_ok(time.time() - start)


def wait_gcs_object(bucket, path):
    # Attente active jusqu'à ce que GCS montre l'objet (cohérence éventuelle)
    print("attente disponibilité GCS...", end="", flush=True)
    for _ in range(10):
        res = subprocess.run(
            ["gsutil", "stat", f"gs://{bucket}/{path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if res.returncode == 0:
            print(" OK")
            return
        print(".", end="", flush=True)
        time.sleep(1)

    print("\nObjet introuvable après délai.")
    exit(1)


def get_last_job_id(region):
    # Retourne le dernier job Vertex AI créé dans la région
    output = run_cmd([
        "gcloud", "ai", "custom-jobs", "list",
        "--region", region,
        "--format=value(name)"
    ], capture=True)

    jobs = output.splitlines()
    if not jobs:
        print(f"{RED}Aucun job trouvé{RESET}")
        exit(1)

    return jobs[0]


def stream_vertex_logs(job_id, region):
    # Affiche les logs Vertex AI en streaming et s'arrête automatiquement à la fin du job
    print(f"\nStreaming logs pour : {job_id}\n")

    proc = subprocess.Popen(
        ["gcloud", "ai", "custom-jobs", "stream-logs", job_id, "--region", region],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in proc.stdout:
        print(line, end="")

        # Détection fin du job dans les logs Vertex AI
        if "completed" in line.lower() or "succeeded" in line.lower():
            print("\nFin d'entraînement détectée — arrêt du streaming.")
            proc.terminate()
            break


    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    # Vérification finale de l'état du job
    status = run_cmd([
        "gcloud", "ai", "custom-jobs", "describe",
        job_id, "--region", region,
        "--format=value(state)"
    ], capture=True)

    print(f"\nÉtat final du job : {status}")

    if "SUCCEEDED" in status:
        print(f"{GREEN}Training terminé avec succès.{RESET}")
    else:
        print(f"{RED}Le job a échoué : {status}{RESET}")
        exit(1)


def step_package():
    # Construction de l’archive src.tar.gz pour Vertex AI
    log_step("package", "génération du package")
    run_cmd(["make", "package"])


def step_upload():
    # Upload du package dans le bucket GCS
    log_step("upload", "envoi du package dans GCS")
    run_cmd(["make", "upload"])
    wait_gcs_object("bucket-crypto-data", "code/src.tar.gz")


def step_fetch():
    # Téléchargement OHLCV, préparation dataset, upload parquet
    log_step("fetch", "fetch + preprocessing + upload parquet")
    run_cmd(["python3", "src/fetch_and_train.py"])

    # Récupération du RUN_ID généré par fetch
    with open("last_run_id.txt", "r") as f:
        rid = f.read().strip()

    print(f"run_id détecté : {rid}")
    return rid


def step_train(run_id):
    # Exécution du job Vertex AI avec le bon RUN_ID
    log_step("train", f"lancement du job Vertex AI (RUN_ID={run_id})")
    run_cmd(["make", "train", f"RUN_ID={run_id}"])

    job_id = get_last_job_id("us-central1")
    print(f"Job créé : {job_id}")

    stream_vertex_logs(job_id, "us-central1")


def step_visual(run_id):
    # Visualisation locale du modèle entraîné (graphs + metrics)
    log_step("visual", "visualisation locale du modèle")
    run_cmd(["make", f"RUN_ID={run_id}", "visual"])


def run_full():
    # Pipeline complet exécuté en un clic
    start = time.time()

    step_package()
    step_upload()
    run_id = step_fetch()
    step_train(run_id)
    step_visual(run_id)

    print(f"{GREEN}pipeline terminé en {time.time() - start:.2f} secondes{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--run_id", required=False)
    args = parser.parse_args()

    if args.full:
        run_full()
    else:
        # Execution manuelle étape par étape
        if args.package:
            step_package()
        if args.upload:
            step_upload()
        if args.train:
            with open("last_run_id.txt", "r") as f:
                rid = f.read().strip()
            step_train(rid)
        if args.visual:
            if not args.run_id:
                print(RED + "Erreur: --run_id est requis pour --visual" + RESET)
                exit(1)
            step_visual(args.run_id)
