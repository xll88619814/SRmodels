#! /bin/bash

###########
### Prepare Resnet50 Model
###########
#rm -f data.tar.gz
#wget http://10.113.3.3/data/example_c++/data.tar.gz
#tar -xzvf data.tar.gz

###########
### Build C++ Resnet50 Model
###########

IGIE_HOME=`python3 -c 'import os; import tvm; print(os.path.dirname(tvm.__file__))'`
if [[ -d "/usr/local/corex-3.2.1/lib64/python3/dist-packages/tvm" ]]; then
    IGIE_HOME=/usr/local/corex-3.2.1/lib64/python3/dist-packages/tvm
fi
echo "IGIE_HOME=${IGIE_HOME}"

export LD_LIBRARY_PATH=${IGIE_HOME}:$LD_LIBRARY_PATH

echo "Build the libraries.."
make clean
make IGIE_ROOT=${IGIE_HOME}

if [ $? -ne 0 ];then
  echo "compile faied!"
  exit 1
fi


###########
### Run Resnet50 Model
###########
# echo "preprocess input data"
# python3 preprocess.py data/ILSVRC2012_val_00000293.JPEG data/ILSVRC2012_val_00000293.txt
# if [[ $? -ne 0 ]];then
#   echo "gen input data failed!"
#   exit 1
# fi

echo "run deploy"
# ./deploy data/ILSVRC2012_val_00000293.txt data/ILSVRC2012_val_00000293_out.txt igie
# ./deploy_profiling_time data/ILSVRC2012_val_00000293.txt data/ILSVRC2012_val_00000293_out.txt igie
./deploy igie
./deploy_profiling_time igie

# echo "process output data"
# python3 postprocess.py data/ILSVRC2012_val_00000293_out.txt
