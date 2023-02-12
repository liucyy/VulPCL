# VulPCL

# Linux Kernel dataset collection

cd crawler

run the following commands to collect: (1) vulnerability commits of Linux Kernel from github advisory; (2) source code of functions from "https://git.kernel.org"; (3) patch files of vulnerability functions  from "https://git.kernel.org".

python get_data_links.py

python get_c_code.py

python get_diff_links.py

python get_diff_files.py

run the following commands to extrac single functions from colleted source code functions.

sh runner_0

# Data pre-processing

cd data_preprocessing

run the following command to preprocess the FFmpeg and QEMU datasets.

python preprocessing.py

# Features extraction

run the following command to extract and embed graph-based representations from CPAGs.

python adc_features_extracting.py

run the following command to extract and embed sequence-based representations from FCDSs.

python cd_features_extracting.py

# For RQ1: vulnerability prediction

cd vul_prediction

run the following command to extract and embed graph-based representations from CPAGs.

python adc_features_extracting.py

run the following command to extract and embed sequence-based representations from FCDSs.

python cd_features_extracting.py

run the following command to train our VulPVL. Where "project_name" could be FFmpeg, qemu, FFmpeg_qemu, linux

python codebert_blstm.py --p project_name

# For RQ2: vulnerability categorization

cd vul_categorization

run the following command to extract and embed graph-based representations from CPAGs.

python adc_features_extracting.py

run the following command to extract and embed sequence-based representations from FCDSs.

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

run the following command to extract and embed graph-based representations from CPAGs.

python adc_features_extracting.py

run the following command to extract and embed sequence-based representations from FCDSs.

python cd_features_extracting.py

run the following command to train our VulPVL.

python codebert_blstm.py --p big_vul