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

# data pre-processing

run the following command to extract and embed graph-based representations from CPAGs.

python adc_features_extracting.py

run the following command to extract and embed sequence-based representations from FCDSs.

python cd_features_extracting.py

run the following command to tokenize the extracted features.

python tokenization.py

run the following command to train our VulPVL. Where "project_name" could be FFmpeg, qemu, FFmpeg_qemu, linux, or big_vul.

python python codebert_blstm.py --p project_name