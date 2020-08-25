import subprocess
import os
from jiwer import wer
import matplotlib.pyplot as plt

alphas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
errors = []

for alpha in alphas:
    count = 0
    sum_error = 0
    for file in os.listdir('../../../datasets/callHome/final/split_audio'):
        command = 'python3 DeepSpeech.py --one_shot_infer ../../../datasets/callHome/final/split_audio/' + file + ' --checkpoint_dir ../deepspeech-0.8.0-checkpoint/deepspeech-0.8.0-checkpoint/'
        print(command)
        text_file = file.split('.')[0] + '.txt'
        text_file = '../../../datasets/callHome/final/split_text/'+text_file
        print(text_file)
        if not os.path.isfile(text_file):
            print("no text file")
            continue
        output = subprocess.getoutput(command)
        prediction = (output.split('\n')[-1])
        ground_truth = (open(text_file).read().replace('\n', ' ')).lower()
        print(prediction)
        print(ground_truth)
        count += 1
        print(wer(ground_truth, prediction))
        sum_error += wer(ground_truth, prediction)
    errors.append(sum_error / count)
plt.plot(alphas, errors)