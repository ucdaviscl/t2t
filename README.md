T2T: Tensor2Tensor
====== 
Sequence-to-sequence model for text simplification.

  - text_simplification.py
  - text_simplification_characters.py


### Docker Installation

Requires [TensorFlow](https://www.tensorflow.org/install/) to run Tensor2Tensor.

```
# Launch the latest TensorFlow GPU binary image in a Docker container
$ nvidia-docker run -it gcr.io/tensorflow/tensorflow:latest-gpu /bin/bash
```

```
# Install with tensorflow-gpu requirement
$ pip install tensor2tensor[tensorflow_gpu]
```

### Adding text_simplification
> T2T's components are registered using a central registration mechanism that enables easily adding new ones and easily swapping amongst them by command-line flag.

Specify the --t2t_usr flag in t2t-trainer.
```
$ t2t-trainer --t2t_usr=/usr/t2t_usr --registry_help
```

```
PROBLEM=text_simplification
MODEL=transformer
HPARAMS=transformer_base_single_gpu

DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
```

### Generate Data
```
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
```

### Train
```
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  > text_simplification.train.output.txt 2>&1
```
Output stored in **text_simplification.train.output.txt**
### Decode
```
DECODE_FILE=$DATA_DIR/normal.valid.txt

BEAM_SIZE=4
ALPHA=0.6
```

```
t2t-decoder \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE
  > text_simplification.decode.output.txt 2>&1
  ```
Output stored in **text_simplification.decode.output.txt**

  
Reference: [tensor2tensor](https://github.com/tensorflow/tensor2tensor)