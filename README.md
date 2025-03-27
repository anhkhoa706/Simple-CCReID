### A Simple Codebase for Clothes-Changing Person Re-identification (Single-GPU Version)

#### Based on: [Clothes-Changing Person Re-identification with RGB Modality Only (CVPR, 2022)](https://arxiv.org/abs/2204.06890)

---

#### ✅ Modified Version Highlights:
- Fully compatible with **single-GPU** setups 
- Removed all **torch.distributed** dependencies 
- Removed **apex** (AMP) requirement 
- Refactored `main.py` to `main_one_gpu.py` for clarity
- Simplified dataloaders and samplers for easier training/testing

---

#### 🛠 Requirements
- Python 3.10+
- PyTorch
- yacs
- h5py
- scipy
- torchvision

> ✅ You no longer need APEX or `torch.distributed` to run this repo.

---

#### 📦 Dataset: CCVID (optional)
- [[BaiduYun]](https://pan.baidu.com/s/1W9yjqxS9qxfPUSu76JpE1g) password: `q0q2`
- [[GoogleDrive]](https://drive.google.com/file/d/1vkZxm5v-aBXa_JEi23MMeW4DgisGtS4W/view?usp=sharing)

You can also use LTCC, PRCC, VC-Clothes, or DeepChange datasets (configurable).

---

#### 🚀 Getting Started

1. Clone this repo and install dependencies:
    ```bash
    git clone https://github.com/your-username/Simple-CCReID.git
    cd Simple-CCReID
    pip install -r requirements.txt
    ```

2. Modify paths in the config:
    Edit `configs/default_img.py` or `configs/default_vid.py`
    ```python
    _C.DATA.ROOT = "/path/to/your/data"
    _C.OUTPUT = "/path/to/save/outputs"
    ```

3. Run training (single GPU):
    ```bash
    python main_one_gpu.py --cfg configs/res50_cels_cal.yaml --dataset prcc --gpu 0
    ```

---

#### 🧪 Evaluation
Model will automatically evaluate every few epochs and print Rank-1 accuracy and other metrics. To run evaluation only:

```bash
python main_one_gpu.py --cfg configs/res50_cels_cal.yaml --dataset prcc --gpu 0 --eval
```

---

#### 📌 Citation
If you use our code or baseline, please cite:

```bibtex
@inproceedings{gu2022CAL,
    title={Clothes-Changing Person Re-identification with RGB Modality Only},
    author={Gu, Xinqian and Chang, Hong and Ma, Bingpeng and Bai, Shutao and Shan, Shiguang and Chen, Xilin},
    booktitle={CVPR},
    year={2022},
}
```

---

#### 🔗 Related Repositories
- [Simple-ReID](https://github.com/guxinqian/Simple-ReID)
- [fast-reid](https://github.com/JDAI-CV/fast-reid)
- [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
- [Pytorch ReID](https://github.com/layumi/Person_reID_baseline_pytorch)

---

📌 Maintainer: [Anh Khoa Nguyen]
Feel free to fork, modify, or contribute back!
