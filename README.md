
# Benchmarking Deep Inverse Models over time, and the Neural-Adjoint method

[comment]: <This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345).> 




## Requirements

### Packages needed:

| Package | Version |
|:---------------------------------------------:|:------------------------------------------------------------------:|
| Python | \>=3.7 |
| Pytorch | \>= 1.3.1 |
| Numpy  | \>=1.17.4 |
| Pandas | \>=0.25.3 |
| Tensorboard | \>=2.0.0 |
| Tqdm| \>=4.42.0 |
| Sklearn | \>=0.22.1|
| Matplotlib | \>= 3.1.3|
|[FrEIA](https://github.com/VLL-HD/FrEIA)  | \>=0.2 | 

### Environment
1. The detailed conda environment is packaged in [.yml file](./demo/environment_droplet.yml).
2. Add the [Benchmarking Algorithm folder](./Benchmarking%20Algorithms) as one of the source directory to make utils and Simulated_Dataset folders 
visible runtime
### Benchmarking datasets

The code to generate the benchmarking dataset can be found in [Simulated_Datasets](./Simulated_DataSets)
For meta-material dataset, see above section for detailed description.
 
### Meta-material Neural simulator
As illustrated in the supplementary material, a ensemble neural simulator has been created for the meta-material dataset
with a high accuracy (mean squared error = 6e-5). The training and testing of the meta-material dataset for benchmarking
performance all depend on this neural simulator. 

The neural simulator is shared by state_dict.pth files (there are 5 models which have the same structure but different weights) in Simulated_DataSets/Meta-material Nerual Simulator/ensemble/state_dicts

To generate dataset for meta-material, first unzip the state-dicts under the same folder and run scipt in [folder](./NA).

The reason why the mm data creation is in the NA is that we are using the forward model nature of NA here.
```create_mm_dataset
python predict.py  (The one line says creat_mm_dataset)
```

The generated file called "MM_data_full.csv" is the Meta-material data that you should read for further training and is regarded as the ground truth for next stages. (Some moving of the data file might be needed)

## Training

To train the models in the paper, go to the each folders for algorithms,
 run this command:

```train
python train.py 
```

> This would read the flags.obj file in the 'models' folder to retrieve the hyper-parameters for each of the
> datasets, afterwhich a training process would be initiated to train a network described in Section 4 of main paper.
> 
> The trained models would be placed in the same 'models' folder with training time and checkpoint file produced.
> Currently the 'models' folders contains the trained hyper-parameters 

## Evaluation

To evaluate the models, for each of the algorithms, run this command:
```eval
python evaluate.py
```

> This would read the flags.obj and best_model.pt file from the 'models' folder and write the evaluation data.
> Note that since we are benchmarking and comparing the time-performance trade-off of various networks structures,
> for a single query data point, multiple trails (200) would be run and each trail would get a different inverse 
> solution (except for Tandem model which is deterministic) as depicted in Fig 4 in main paper.
>
## Results

Our model achieves the following performance on :

![Inverse model performance as a function of time](./demo/3.png) 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 