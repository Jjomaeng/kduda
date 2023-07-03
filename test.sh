# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

#!/bin/bash

TEST_ROOT=$1
CONFIG_FILE="/home/cvlab/PycharmProjects/pythonProject/DAFormer/work_dirs/local-basic/230613_1523_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb3_s0_80886/230613_1523_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb3_s0_80886.json"
CHECKPOINT_FILE="/home/cvlab/PycharmProjects/pythonProject/DAFormer/work_dirs/local-basic/230613_1523_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb3_s0_80886/latest.pth"
SHOW_DIR="${TEST_ROOT}/preds/"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${SHOW_DIR} --opacity 1
