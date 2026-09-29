# Fashion-MNIST ANN Pipeline

A reproducible Machine Learning pipeline for classifying Fashion-MNIST images using an Artificial Neural Network (ANN). The project combines model development with **Git** and **DVC (Data Version Control)** to provide version control for source code, datasets, model artifacts, experiments, parameters, and evaluation results.

The main purpose of this project is not only to train a neural network, but also to demonstrate a structured and reproducible MLOps workflow. The complete process — from downloading the dataset to preprocessing, training, and evaluation — is represented as a DVC pipeline.

---

## Project Overview

Fashion-MNIST is an image classification dataset containing grayscale images of different clothing categories. Each image has a size of **28 × 28 pixels**, and the dataset contains **10 different classes**.

This project builds a simple Artificial Neural Network using TensorFlow/Keras to classify these images.

The project follows the following workflow:

```text
Fashion-MNIST Dataset
        |
        v
   Data Preparation
        |
        v
   Data Preprocessing
        |
        v
     ANN Training
        |
        v
  Model Evaluation
        |
        v
Metrics + Confusion Matrix
```

The workflow is automated using DVC, allowing the required pipeline stages to be reproduced whenever source code, parameters, or dependencies change.

---

## Objectives

The main objectives of this project are:

- Build an ANN model for Fashion-MNIST image classification.
- Organize ML code into separate preparation, preprocessing, training, and evaluation stages.
- Use Git for source code version control.
- Use DVC for dataset, model, and experiment reproducibility.
- Store large ML artifacts using DVC instead of Git.
- Configure Google Drive as remote DVC storage.
- Define a reproducible ML pipeline using `dvc.yaml`.
- Manage experiment parameters using `params.yaml`.
- Track pipeline state using `dvc.lock`.
- Evaluate the trained model using accuracy, loss, and a confusion matrix.
- Demonstrate experimentation using Git branches and tags.
- Demonstrate merge-conflict handling in an ML/DVC workflow.

---

## Project Structure

```text
fashion-ann-pipeline/
|
|-- .dvc/
|   `-- config
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- metrics/
|   `-- confusion_matrix.png
|
|-- models/
|   |-- model.h5
|   `-- history.csv
|
|-- src/
|   |-- prepare.py
|   |-- preprocess.py
|   |-- train.py
|   `-- evaluate.py
|
|-- .dvcignore
|-- .gitignore
|-- dvc.yaml
|-- dvc.lock
|-- params.yaml
|-- metrics.json
`-- README.md
```

Large generated artifacts such as datasets and trained models are managed through DVC and are not intended to be stored directly in Git.

---

## DVC Pipeline

The project contains four main DVC stages:

```text
prepare
   |
   v
preprocess
   |
   v
train
   |
   v
evaluate
```

The complete pipeline can be displayed using:

```bash
dvc dag
```

and reproduced using:

```bash
dvc repro
```

DVC determines which stages need to run based on changes to dependencies, parameters, and outputs. Unchanged stages can therefore be skipped automatically.

---

## 1. Data Preparation

The first pipeline stage is implemented in:

```text
src/prepare.py
```

It downloads Fashion-MNIST using TensorFlow/Keras:

```python
keras.datasets.fashion_mnist.load_data()
```

The original training and test datasets are saved as NumPy arrays inside:

```text
data/raw/
```

The generated files include:

```text
X_train.npy
y_train.npy
X_test.npy
y_test.npy
```

This stage is represented in `dvc.yaml` as the `prepare` stage.

---

## 2. Data Preprocessing

The preprocessing stage is implemented in:

```text
src/preprocess.py
```

It loads the raw Fashion-MNIST arrays and prepares them for model training.

The current preprocessing implementation scales image values using:

```python
X_train = X_train.astype('float32') / 255.0 * 0.9
X_test = X_test.astype('float32') / 255.0 * 0.9
```

The training dataset is then divided into training and validation sets using `train_test_split`.

The split is controlled through `params.yaml`:

```yaml
preprocess:
  test_size: 0.2
  seed: 101
```

Therefore:

- `20%` of the original training data is used for validation.
- Random seed `101` is used to make the split reproducible.

The processed arrays are stored inside:

```text
data/processed/
```

including training, validation, and test data.

---

## 3. ANN Model Training

The training stage is implemented in:

```text
src/train.py
```

The model is created using TensorFlow/Keras.

### Network Architecture

```text
Input Image (28 × 28)
        |
        v
Flatten Layer
        |
        v
Dense Layer (128 neurons, ReLU)
        |
        v
Dropout Layer
        |
        v
Dense Output Layer (10 neurons, Softmax)
```

The `Flatten` layer converts each 28 × 28 image into a one-dimensional representation.

The hidden layer contains **128 neurons** and uses the **ReLU** activation function.

A dropout layer is included to reduce overfitting.

The final layer contains **10 neurons**, corresponding to the 10 Fashion-MNIST classes, and uses the **Softmax** activation function.

### Optimizer and Loss

The model uses:

```text
Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Metric: Accuracy
```

The trained model is saved as:

```text
models/model.h5
```

Training history is saved as:

```text
models/history.csv
```

---

## Model Parameters

Training and preprocessing parameters are stored separately in:

```text
params.yaml
```

Current parameters are:

```yaml
preprocess:
  test_size: 0.2
  seed: 101

train:
  epochs: 10
  learning_rate: 0.001
  dropout_rate: 0.2
  batch_size: 32
```

Keeping parameters outside the source code makes experiments easier to reproduce and compare.

DVC monitors these parameters through the `params` sections defined in `dvc.yaml`.

---

## 4. Model Evaluation

The evaluation stage is implemented in:

```text
src/evaluate.py
```

The script loads the trained model and evaluates it against the Fashion-MNIST test dataset.

It calculates:

- Test accuracy
- Test loss
- Class predictions
- Confusion matrix

The numerical evaluation results are stored in:

```text
metrics.json
```

The confusion matrix is stored as:

```text
metrics/confusion_matrix.png
```

The metrics can be displayed directly through DVC:

```bash
dvc metrics show
```

A final pipeline run produced approximately:

```text
Test Accuracy: 0.8796
Test Loss:     0.33808
```

This corresponds to approximately **87.96% test accuracy**.

Because neural-network training involves numerical and initialization-related variability, exact results may differ slightly when the model is retrained.

---

## DVC and Data Version Control

Git works well for source code and small text files but is not designed for repeatedly storing large ML datasets and trained model files.

DVC is therefore used alongside Git.

### Git tracks

Git is used to version files such as:

```text
src/*.py
dvc.yaml
dvc.lock
params.yaml
README.md
.gitignore
```

### DVC manages

DVC manages generated ML artifacts such as:

```text
data/raw/
data/processed/
models/
metrics.json
metrics/confusion_matrix.png
```

This keeps the Git repository lightweight while still allowing the ML state associated with a particular version of the project to be reproduced.

---

## DVC Lock File

The project contains:

```text
dvc.lock
```

The lock file records the exact state of pipeline dependencies, parameters, and outputs.

When a dependency or parameter changes and the pipeline is reproduced, DVC updates the corresponding information in `dvc.lock`.

This helps connect a particular Git revision with the corresponding state of the ML pipeline.

---

## DVC Remote Storage

A Google Drive folder is configured as the DVC remote.

Large artifacts can be uploaded using:

```bash
dvc push
```

and restored using:

```bash
dvc pull
```

The DVC remote allows large datasets and model artifacts to remain outside the Git repository while still being recoverable when reproducing the project.

Authentication credentials are kept outside the Git-tracked project configuration and should never be committed to the repository.

---

## Reproducing the Pipeline

After obtaining the repository and installing the required dependencies, DVC-managed artifacts can be restored using:

```bash
dvc pull
```

The pipeline can then be checked with:

```bash
dvc status
```

and reproduced using:

```bash
dvc repro
```

If no relevant dependency, source file, or parameter has changed, DVC may skip the corresponding stages.

The pipeline graph can be viewed with:

```bash
dvc dag
```

Metrics can be displayed with:

```bash
dvc metrics show
```

---

## Experimentation and Versioning

Git tags and branches are used to preserve different stages and experiments performed during development.

### Tags

The repository contains experimental/version tags including:

```text
v1
v2
e2
```

`v1` represents the reproducible pipeline baseline.

`v2` represents an experiment in which the number of training epochs was changed from **10 to 12**.

`e2` represents a preprocessing experiment involving modified normalization.

These versions demonstrate how Git and DVC can be combined to preserve changes to both code/configuration and ML artifacts.

---

## Branching Workflow

The project uses multiple Git branches:

```text
main
dev
teammate-sim
```

### `main`

Contains the integrated project and resolved workflow.

### `dev`

Used during development and parameter experimentation, including the 12-epoch experiment associated with `v2`.

### `teammate-sim`

Used to simulate a teammate making an independent preprocessing change.

This branch was used to demonstrate a realistic Git merge-conflict scenario involving ML preprocessing and DVC metadata.

The conflict was resolved on `main`, after which the DVC pipeline was reproduced and verified.

---

## Git and DVC Workflow

The project demonstrates the complementary roles of Git and DVC.

```text
             Git
              |
      Source Code + Config
              |
              v
         dvc.yaml
              |
              v
     Reproducible Pipeline
              |
              v
    Data / Models / Metrics
              |
              v
             DVC
              |
              v
     Google Drive Remote
```

Git records how the project code and configuration evolve, while DVC handles the large data and ML artifacts generated by that code.

Together they provide a reproducible version-controlled machine-learning workflow.

---

## Useful Commands

### Check Git state

```bash
git status
```

### View Git history

```bash
git log --oneline --graph --all --decorate
```

### Check DVC pipeline status

```bash
dvc status
```

### Display pipeline graph

```bash
dvc dag
```

### Reproduce changed pipeline stages

```bash
dvc repro
```

### Display evaluation metrics

```bash
dvc metrics show
```

### Upload DVC artifacts

```bash
dvc push
```

### Download DVC artifacts

```bash
dvc pull
```

### Restore DVC-managed workspace files

```bash
dvc checkout
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- PyYAML
- Git
- GitHub
- DVC
- Google Drive

---

## Key Learning Outcomes

This project demonstrates several important MLOps concepts:

- Separating ML workflows into reproducible stages
- Version controlling source code with Git
- Managing large ML artifacts using DVC
- Using remote storage for datasets and models
- Tracking experiment parameters
- Reproducing only affected pipeline stages
- Recording evaluation metrics
- Using Git branches for parallel development
- Resolving merge conflicts in an ML project
- Connecting code versions with corresponding ML artifacts

---

## Author

**Umer Karamat**  
