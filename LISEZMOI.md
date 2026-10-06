# InSpatio-World 1.5 — guide en français

Ce dépôt contient le code d'[InSpatio-World 1.5](https://github.com/inspatio/inspatio-world-v1.5)
(licence Apache-2.0, © InSpatio Team), copié tel quel à partir du commit `dd3561f`.
Le guide d'origine, en anglais, est dans [README.md](README.md).

Le modèle prend **une photo, quatre photos ou une vidéo** et génère une vidéo de la
même scène vue depuis une autre trajectoire de caméra.

## Ce qu'il te faut

| | Minimum conseillé |
|---|---|
| Carte graphique | NVIDIA, **16 Go de mémoire vidéo au moins, 24 Go ou plus conseillés** (non testé en dessous) |
| Système | **Linux** (Ubuntu 22.04 / 24.04). Sous Windows : passer par **WSL2** (voir plus bas) |
| Pilote NVIDIA | récent, compatible CUDA 12.6 (`nvidia-smi` doit afficher « CUDA Version: 12.6 » ou plus) |
| Disque | **~50 Go libres** (modèles + conversion + résultats) |
| Logiciels | `git`, `git-lfs`, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) |

Avec moins de 40 Go de mémoire vidéo, le programme garde automatiquement l'encodeur de texte
(le plus gros morceau) en mémoire vive et ne l'envoie sur la carte graphique qu'au besoin :
prévois donc aussi **32 Go de RAM** si possible.

### Windows : installer WSL2

Le projet utilise des scripts Linux (`bash`, `nvidia-smi`). Sous Windows 10/11 :

1. Mets à jour ton pilote NVIDIA (il gère CUDA dans WSL automatiquement).
2. Dans PowerShell **en administrateur** : `wsl --install -d Ubuntu-24.04`, puis redémarre.
3. Ouvre « Ubuntu » depuis le menu Démarrer et fais toute la suite dans ce terminal.
4. Vérifie que la carte est vue : `nvidia-smi`.

Ne réinstalle **pas** de pilote NVIDIA à l'intérieur d'Ubuntu/WSL.

## Installation (une seule fois)

```bash
# 1. Outils de base
sudo apt update && sudo apt install -y git git-lfs
git lfs install

# 2. Récupérer ce dépôt
git clone https://github.com/MaxiJoun/In-spatio.git
cd In-spatio

# 3. Environnement Python (télécharge PyTorch + CUDA, plusieurs Go)
conda env create -f environment.yml
conda activate inspatio_world_test
python -m pip install --no-deps depth-anything-3==0.1.1

# 4. Télécharger les modèles (~25 Go, ça peut prendre longtemps)
bash pipeline/download.sh
```

Si tu as une carte récente de type H100, tu peux installer FlashAttention-3 pour aller plus vite ;
sinon le programme utilise l'attention standard de PyTorch, ça marche sans.

## Lancer les exemples

```bash
conda activate inspatio_world_test
cd In-spatio

# Un seul exemple (le plus rapide pour tester) :
bash run_inference.sh --scene_dir examples/image_example_00

# Les six exemples fournis :
bash run_example.sh
```

Au premier lancement, l'encodeur de texte est converti au format `.safetensors` (une fois pour toutes).

Les résultats arrivent dans `output/<nom_de_la_scène>/` :

| Fichier | Contenu |
|---|---|
| `pred.mp4` | **la vidéo générée** — c'est le résultat |
| `source.mp4` | l'entrée d'origine |
| `render.mp4` | rendu brut de la nouvelle caméra (avec des trous) |
| `mask.mp4` | zones que le modèle a dû inventer |

## Utiliser ta propre vidéo

```bash
bash run_inference.sh --video /chemin/vers/ma_video.mp4 \
  --prompt "Description de la scène en anglais" \
  --target_traj /chemin/vers/target_tcw.txt
```

- La vidéo est redimensionnée en 832×480 automatiquement.
- La profondeur et les caméras d'origine sont estimées automatiquement (Depth-Anything-3).
- `target_tcw.txt` décrit la caméra voulue : **une ligne par image**, 16 nombres (matrice 4×4
  monde→caméra, convention OpenCV, lue ligne par ligne), dans le même repère que la caméra
  estimée de la première image. Le plus simple est de partir d'un des fichiers
  `examples/video_example_*/input/target_tcw.txt` (même nombre de lignes que d'images dans ta vidéo).

Pour tes propres photos (1 ou 4 images), il faut fournir en plus les cartes de profondeur et
les caméras : voir [examples/README.md](examples/README.md).

## En cas de problème

| Message | Solution |
|---|---|
| `nvidia-smi: command not found` / `No GPU is visible` | pilote NVIDIA absent ou, sous Windows, commandes lancées hors de WSL |
| `CUDA out of memory` | ferme les autres programmes qui utilisent la carte ; essaie d'abord `image_example_00` |
| erreur pendant `git lfs pull` | `git lfs install` puis relance `bash pipeline/download.sh` |
| plusieurs cartes graphiques | choisis-en une avec `--gpu 0` (ou 1, 2…) |

## Licence

Code sous licence Apache-2.0 (voir [LICENSE](LICENSE)), auteurs : InSpatio Team.
Les dépendances (Depth-Anything-3, Wan2.1…) ont leurs propres licences.
Seul ce fichier `LISEZMOI.md` et la ligne d'en-tête de `README.md` ont été ajoutés.
