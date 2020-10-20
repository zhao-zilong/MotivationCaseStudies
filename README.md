# background_research
Offline training with dirty label data, varying the noisy level from 0% to 100%. For IoT attack (thermostat) dataset and Cluster failure detection (task) dataset, all the experiments on each algorithm are repeated 10 times. For face dataset with MLP algorithm, we also run 10 times the experiments, but for VGG algorithm, we repeat only 3 times experiments due to the complexity of training. Another exception is the experiments of nearest centroid algorithm on Cluster failure detection (task) dataset, when we run the experiments, the results vary a lot from each run, so we run 100 times the experiments, and report the averaged result in the end.

All the intermediate results and final results are shown in assessement_*.ipynb files, for face dataset, use two separated .ipynb files.
If any reader wants to reproduce the results, the datasets are uploaded on [google drive.](https://drive.google.com/file/d/1VKeYKg_0jsi4Vb6GFTaHuCLHWqwO3eZ6/view?usp=sharing)
