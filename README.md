# SF-IQA
SF-IQA: Quality and Similarity Integration for AI Generated Image Quality Assessment
> [**SF-IQA**](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Yu_SF-IQA_Quality_and_Similarity_Integration_for_AI_Generated_Image_Quality_CVPRW_2024_paper.pdf), Zihao Yu, Fengbin Guan, Yiting Lu, Xin Li, Zhibo Chen

> Accepted by CVPR2024 Workshop 

![image](https://github.com/Travois/SF-IQA/blob/main/method.png)

## Overview

This README provides essential information for setting up and running the program associated with this project. Please follow the instructions carefully to ensure the environment is correctly prepared and the program runs as expected.

## Environment Setup

Different environment may induce possible fluctuation of performance.

Before executing the program, it's crucial to prepare the environment to meet the software dependencies required.

### Download the files to the folder

Download [weights_dict.pt](https://rec.ustc.edu.cn/share/d4b86c90-3e09-11ef-a8bf-6358f51f862b) to SF-IQA/pretrained

Download [model.safetensors](https://rec.ustc.edu.cn/share/8d69ccd0-3e0a-11ef-aa31-2b14741a59d0) to SF-IQA/pretrained/PickScore_v1

### Creating a New Conda Environment

To create a new environment with all the necessary dependencies, run the following command:

```bash
conda env create -f environment.yml
```
This command will set up a new conda environment based on the specifications in the environment.yml file.

### Important Dependency Versions
Please ensure that the following dependencies are installed with the correct versions to avoid any compatibility issues:
```
timm==0.6.13
transformers==4.38.2
```
Using different versions of these packages may result in errors during program execution.

## Configuration
Before running the program, you need to configure the test.sh script with the correct paths for your datasets and output locations:

datasets_path: Replace this with the path to your Testing Images folder.
Example:/AIGCQA-30K-Image/test

info_path: Replace this with the file path to your Testing Prompts file.
Example:/AIGCQA-30K-Image/info_test.xlsx'

output_path: Replace this with the folder path where the output.txt file will be saved.
Example:./outfolder

<details>
<summary>AIGCQA-30K-Image</summary>


 ```
AIGCQA-30K-Image
    |--test
    |    |  DALLE2_0001.png   
    |    |  ... (all the images should be here)     
    |--info_test.xlsx
```
</details>


## Execution
### Using a GPU
If you're running the program on a GPU, you can specify the GPU to use with the following options:

```bash
--use_gpu --gpu_id 0
```
This program is designed to run on a single GPU. Adjust the --gpu_id value according to the GPU you wish to use.

### Using a CPU
If you're running the program on a CPU, please ensure to remove the --use_gpu --gpu_id 0 from the script.

### Adjusting Batch Size
You can adjust the batch size for the program execution by using the --batch_size option. This allows for flexibility based on your system's capabilities.

### Progress Display
To display the program's progress, you can use the --use_tqdm option. The program will show progress bars four times before completion.

### Running the Program
After adjusting the test.sh file with your specific configurations, execute the program by running:

```
bash test.sh
```
This will start the program using the settings you've configured.


## Pretrained models
The pretrain models are provided from [swinv2 tiny patch4 window16 256](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth) and [PickScore v1](https://huggingface.co/yuvalkirstain/PickScore_v1). 

## Acknowledgement
This code is borrowed parts from [SAMA](https://github.com/Sissuire/SAMA) and  [TReS](https://github.com/isalirezag/TReS). 

## Citation
If you find this work useful for your research, please cite our paper:
```
@InProceedings{Yu_2024_CVPR,
    author    = {Yu, Zihao and Guan, Fengbin and Lu, Yiting and Li, Xin and Chen, Zhibo},
    title     = {SF-IQA: Quality and Similarity Integration for AI Generated Image Quality Assessment},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {6692-6701}
}
```

## Note
For any issues or further assistance, please refer to the project's documentation or contact the support team.
