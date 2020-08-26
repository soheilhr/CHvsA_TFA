# CHvsA_TFA: Taskforce A , ML model for gait classification of Adults vs Childs

Train.py
A function to train a fully convolutional neural net on complex spectrogram data

Arguments:
--train_data_path: path to train data
--model_save_path: path to save model
--exp_type: experiment type, sim_v0 or exp_v0
--model_name_tag: model name tag
--on_gpu: gpu number

Resulting model will be dated and saved in the corresponding path

Sample use case:
python train.py --model_name_tag 'test0'

Test.py
A function to test a fully convolutional neural net on complex spectrogram data

Arguments:
--test_data_path: path to test data
--model_path: path to the model
--report_path: path to the report
--exp_type: experiment type, sim_v0 or exp_v0
--report_name_tag: report name tag
--on_gpu: gpu number

Sample use case:
python train.py --report_name_tag 'test0'













