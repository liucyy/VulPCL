![image](https://github.com/user-attachments/assets/a009e182-7073-4356-bbeb-e9e544fd49a1)# Making vulnerability prediction more practical: Prediction, categorization, and localization

VulPCL is a BLSTM and CodeBERT based approach to perform vulnerability prediction, categorization, and localization automatically within a framework. This repo is the artifact for paper [Making vulnerability prediction more practical: Prediction, categorization, and localization](https://www.sciencedirect.com/science/article/abs/pii/S0950584924000636), which has been accepted by IST'24.

# Linux Kernel dataset collection

cd crawler

run the following commands to collect: (1) vulnerability commits of Linux Kernel from github advisory; (2) source code of functions from: https://git.kernel.org; (3) patch files of vulnerability functions  from: https://git.kernel.org.

python get_data_links.py

python get_c_code.py

python get_diff_links.py

python get_diff_files.py

run the following commands to extrac single functions from colleted source code functions.

sh runner_0

# Datasets Downloading

the source code of: (1) FFmpeg and QEMU datasets are available at: https://sites.google.com/view/devign; (2) our collected Linux Kernel dataset is available: https://www.mediafire.com/file/ux2x1m7i2kwom7z/dataset.zip/file; (3) Big-Vul dataset is available at: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset.

# Data pre-processing

cd data_preprocessing

run the following command to preprocess the FFmpeg and QEMU datasets.

python preprocessing.py

create three directories: './vul_prediction/data/code', './categorization/data/code', and './vul_localization/data/code', then put the filtered source code of each dataset into corresponding directory according to the RQs.

# For RQ1: vulnerability prediction

cd vul_prediction

run the following command to extract and embed graph-based features from CPAGs.

python adc_features_extracting.py

run the following command to extract and embed sequence-based features from FCDSs.

python cd_features_extracting.py

run the following command to train our VulPVL. Where "project_name" could be FFmpeg, qemu, FFmpeg_qemu, linux

python codebert_blstm.py --p project_name

# For RQ2: vulnerability categorization

cd vul_categorization

run the following command to extract and embed graph-based features from CPAGs.

python adc_features_extracting.py

run the following command to extract and embed sequence-based features from FCDSs.

python cd_features_extracting.py

run the following command to perform 10-vulnerabilities labeling.

python vul_files_label.py

run the following command to train our VulPVL.

python codebert_blstm.py --p linux

run the following command to train CodeBERT.

python codebert.py --p linux

run the following command to train CNN.

python CNN.py --p linux

run the following command to train GRU.

python GRU.py --p linux

# For RQ3: vulnerability localization

cd vul_localization

run the following command to extract and embed graph-based features from CPAGs.

python adc_features_extracting.py

run the following command to extract and embed sequence-based features from FCDSs.

python cd_features_extracting.py

run the following command to train our VulPVL.

python codebert_blstm.py --p big_vul
