# Project Setup

## Setup Environment

To set up the environment for this project, please follow the steps below:

1. **Create a new conda environment** named `DiffDis`:
    ```bash
    conda create -n DiffDis
    ```

2. **Activate** the newly created environment:
    ```bash
    conda activate DiffDis
    ```

3. **Install PyTorch** and related libraries with CUDA support:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

4. **Install additional required Python packages** from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Change Directory

Navigate to the `code` directory where the project scripts are located:
```bash
cd code
```

## Download Data

To download the necessary data for the project, execute the following script:
```bash
bash download_data.sh
```

## Download and Extract CC3M Dataset

To download and extract the CC3M dataset, run the following scripts in order:

1. **Download the CC3M dataset**:
    ```bash
    bash download_cc3m_dataset.sh
    ```

2. **Extract the downloaded CC3M dataset**:
    ```bash
    bash extract_cc3m_dataset.sh
    ```