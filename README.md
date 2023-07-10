<div align="center">
<p align="center">
  <img src="./assets/intro.jpg" width="50%" height="50%" />
</p>
</div>

<div align="center">
<h1>Explicit Syntactic Guidance for Neural Text Generation</h1>
</div>

<div align="center">
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-CC%20BY%204.0-green.svg" alt="License">
<img src="https://img.shields.io/github/stars/yafuly/SyntacticGen?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/yafuly/SyntacticGen?color=red" alt="Issues">




<!-- **Authors:** -->
<br>

_**Yafu Li<sup>†</sup><sup>‡</sup>, Leyang Cui<sup>¶</sup>, Jianhao Yan<sup>†</sup><sup>‡</sup>, Yongjing Yin<sup>†</sup><sup>‡</sup>,<br>**_

_**Wei Bi<sup>¶</sup>, Shuming Shi<sup>¶</sup>, Yue Zhang<sup>‡</sup><br>**_


<!-- **Affiliations:** -->


_<sup>†</sup> Zhejiang University,
<sup>‡</sup> Westlake University,
<sup>¶</sup> Tencent AI Lab_


A syntax-guided generation schema that generates the sequence guided by a constituency parse tree in a top-down direction ([See paper for details](https://arxiv.org/abs/2306.11485)).
</div>



## Environment Setup
To set up the environment, clone the project and run the following script:

```bash
bash setup_env.sh # run under SyntacticGen directory
```

## Toy Example
To demonstrate the process, we have prepared a toy example that includes data preparation, model training, and model inference. Please ensure to specify the "PROJECT_PATH" in each script according to your actual project path.
```bash

# Set the PROJECT_PATH variable to your directory

PROJECT_PATH=$YOUR_DIRECTORY

# Data Preparation: This script builds training triplets using source text data, target text data, and parsing results.

bash $PROJECT_PATH/shell/prepare_data.sh

# Model Training: Train the neural decoder on the training triplets.

bash $PROJECT_PATH/shell/train.sh

# Model Inference: Use the trained model for inference. Structural beam search is enabled if beam size is larger than 1.

bash $PROJECT_PATH/shell/infer.sh

```

We provide the model for paraphrase generation in our paper at Google Drive.

## Citation
```bibtex
@misc{li2023explicit,
      title={Explicit Syntactic Guidance for Neural Text Generation}, 
      author={Yafu Li and Leyang Cui and Jianhao Yan and Yongjing Yin and Wei Bi and Shuming Shi and Yue Zhang},
      year={2023},
      eprint={2306.11485},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
