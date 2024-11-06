@echo off
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2" /M
setx PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;%PATH%" /M
setx PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp;%PATH%" /M
setx NVCUDASAMPLES_ROOT "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.2" /M
setx NVCUDASAMPLES11_2_ROOT "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.2" /M
echo Permanently switched to CUDA 11.2 with CUDA sample paths set
