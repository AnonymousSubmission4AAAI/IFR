# Learning on Noisy Text Datasets via Iterative Filtering and Reweighting

The sample code for the paper "Learning on Noisy Text Datasets via Iterative Filtering and Reweighting" submitted to AAAI 2023.

# Abstract

 Noisy labels in datasets have always been an essential dilemma in Deep Learning studies. Previous works mainly suffer from two kinds of problems. One is the over-detection problem, which eliminates hard samples and leads to decreased generalization performance. The other is poor performance on high-ratio noisy label datasets since the distribution of label noise is more random. To solve the problem, we adopt the idea of continuously using purified samples to train the model in this paper. Our solution proposes an iterative training strategy for uncertain text datasets by effective filtering and ingenious reweighting processes. The method starts from initial training supported by a small number of high-quality data. A filtering part is performed in every iteration by removing the confident noisy labels based on probability predictions. Then, the filtered samples are reweighted by the training performance on repeated extractions. Empirical results on four text datasets with different ratios of randomly introduced label noise demonstrate that our method can competitively improve training performance.

# Dependencies

The latest version of `torch`, `numpy`, `pytorch_pretrained_bert`, `sklearn`.

# Data Format

**Input(Train, Inaccurate, Test):** `.tsv` or `.csv` file and specifying the data indexes and the label index.
**Output:** one column for the predicted label.

# Run

```
python ifr_main.py --class_list=0,1 \
          --batch=256 \ 
          --epoch = 5 \
          --base_path=<the path contains the BERT pretrained model and the input files.> \
          --train_file=<train file name> \
          --inaccurate_file=<inaccurate file name> \
          --test_file=<test file name> \
          --pad=64 \
          --bert=<BERT dir name> \
          --txt_a=<first text indexes> \
          --txt_b=<second text indexes> \
          --label_idx=<label index> \
          --random_seed=1024 \
          --seperator="\t" \
          --cv_nums=5 \
          --job_name=<log file name> \
          --lang="cn"/"en" \
          --choice_rate=0.3 \
          --choice_num=13 \
          --iteration_threshold=10 \
          --choice_rate_train=0.1 
```
