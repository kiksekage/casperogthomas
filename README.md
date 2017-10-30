# casperogthomas

## How to run models:
```
python3 sem_eval.py [perceptron/nearest_centroid/random_forest] [en/es/ar] [17/18]
```
With cwd being "testkode". NOTE: if models are run on 17 data, the models will be trained on english training and test data from 17. 

### How to run eval scripts:
```
./eval.sh [reg/class]
```
With cwd being "testkode"
NOTE: [EmoInt](https://github.com/felipebravom/EmoInt) in the folder outside of the repository is needed to run the eval shell script. Furthermore, the arabic test results in regression can not be evaluated locally.
