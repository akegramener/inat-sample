import argparse
import os
import pickle
import time
import pandas as pd
import numpy as np
import cntk as C
import _cntk_py
import cntk.io.transforms as xforms
from cntk.train.training_session import *
from cntk.logging import *
from cntk.debugging import *
from PIL import Image

parser = argparse.ArgumentParser(description='Image Classification Model')
parser.add_argument('--train', '-t', action='store_true', help='Train the model', default=False)
parser.add_argument('--evaluate', '-e', action='store_true', help='Evaluate the model', default=False)
parser.add_argument('--model', '-m', action='store', help='Model Path')

abs_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(abs_path, 'model')
data_path = os.path.join(abs_path, 'metadata')
model_name = 'resnet34-inat.model'
train_dir = 'data/train/'
test_dir = 'data/validation/'

id2label = pickle.load(open('metadata/id2label', 'rb'))
label2id = pickle.load(open('metadata/label2id', 'rb'))
cat_subcat = pickle.load(open('metadata/cat_subcat', 'rb'))

# model dimensions
image_height = 224
image_width  = 224
num_channels = 3
num_classes  = sum(1 for i in os.listdir(os.path.join(abs_path, 'data', 'train')))
epoch_size = sum(1 for line in open(os.path.join(abs_path, 'metadata', 'train_map.txt')))
print('Number of classes: {}'.format(num_classes))
print('Epoch size: {}'.format(epoch_size))

# Create a minibatch source.
def create_image_mb_source(map_file, train, total_number_of_samples):
    print('Creating source for {}.'.format(map_file))
    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomarea', area_ratio=(0.08, 1.0), aspect_ratio=(0.75, 1), jitter_type='uniratio'), # train uses jitter
            xforms.color(brightness_radius=0.4, contrast_radius=0.4, saturation_radius=0.4)
        ]

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='cubic')
    ]

    # deserializer
    return C.io.MinibatchSource(
        C.io.ImageDeserializer(
            map_file,
            C.io.StreamDefs(features=C.io.StreamDef(field='image', transforms=transforms), # 1st col in mapfile referred to as 'image'
                            labels=C.io.StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize=train,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)


# Create the model.
def resnet_model(name, scaled_input):
    print('Loading Resnet model from {}.'.format(name))
    base_model = C.load_model(os.path.join(model_path, name))
    feature_node = C.logging.find_by_name(base_model, 'features')
    last_node = C.logging.find_by_name(base_model, 'z.x')

    # Clone the desired layers with fixed weights
    cloned_layers = C.combine([last_node.owner]).clone(C.CloneMethod.freeze, {feature_node: C.placeholder(name='features')})
    cloned_out = cloned_layers(scaled_input)

    z = C.layers.GlobalAveragePooling()(cloned_out)
    z = C.layers.Dropout(dropout_rate=0.3, name='d1')(z)

    z = C.layers.Dense(128, activation=C.ops.relu, name='fc1')(cloned_out)
    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = C.layers.Dropout(dropout_rate=0.6, name='d2')(z)

    z = C.layers.Dense(128, activation=C.ops.relu, name='fc2')(cloned_out)
    z = C.layers.BatchNormalization(map_rank=1)(z)
    z = C.layers.Dropout(dropout_rate=0.3, name='d2')(z)

    z = C.layers.Dense(num_classes, activation=None, name='prediction')(z)

    return z

# TODO: Load the trained model and unfreeze the layers
# def load_and_unfreeze_model(name, scaled_input):
#     print('Loading trained model from {}.'.format(os.path.join(model_path, name)))
#     path = os.path.join(model_path, name)
#     if not os.path.exists(path):
#         raise FileNotFoundError('Initial model of phase 1 doesnot exist, please complete the initial steps!')
#     model = C.load_model(path)
#     clone = model.clone(C.CloneMethod.clone)
#     z = clone(scaled_input)
#     return z

# Create the network.
def create_resnet_network():
    print('Creating the network.')
    # Input variables denoting the features and label data
    feature_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))

    # apply model to input
    scaled_input = feature_var - C.constant(114)
    z = resnet_model('ResNet34_ImageNet_CNTK.model', scaled_input)

    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    C.logging.log_number_of_parameters(z)

    return {
        'feature': feature_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'output': z
    }

# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits, warm_up, progress_writers):
    print('Creating the trainer.')
    # Train only the last layers
    lr_schedule = C.learning_rate_schedule([0.01] * 10 + [0.001] * 20 + [0.0001] * 30, unit=C.UnitType.minibatch)
    mm_schedule = C.momentum_schedule(0.9)
    l2_reg_weight = 0.0001

    learner = C.adam(network['output'].parameters,
                  lr_schedule,
                  mm_schedule,
                  l2_regularization_weight=l2_reg_weight,
                  unit_gain=False)

    num_workers = C.distributed.Communicator.num_workers()
    print('Number of workers: {}'.format(num_workers))
    if num_workers > 1:
        parameter_learner = C.train.distributed.data_parallel_distributed_learner(learner, num_quantization_bits=num_quantization_bits)
        trainer = C.Trainer(network['output'], (network['ce'], network['pe']), parameter_learner, progress_writers)
    else:
        trainer = C.Trainer(network['output'], (network['ce'], network['pe']), learner, progress_writers)

    return trainer

# Train and test
def train_model(network, trainer, train_source, test_source, validation_source, minibatch_size, epoch_size, restore, profiling=False):
    print('Training the model.')
    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    # Train all minibatches
    if profiling:
        start_profiler(sync_gpu=True)

    def callback(index, average_error, cv_num_samples, cv_num_minibatches):
        print('Epoch:{}, Validation Error: {}'.format(index, average_error))
        return True

    checkpoint_config = CheckpointConfig(frequency=epoch_size, filename=os.path.join(model_path, "resnet34_cp"), restore=restore)
    # test_config = TestConfig(minibatch_source=test_source, minibatch_size=minibatch_size)
    validation_config = CrossValidationConfig(
        minibatch_source=validation_source,
        frequency=epoch_size,
        minibatch_size=minibatch_size,
        callback=callback)

    start = time.time()
    training_session(
        trainer=trainer,
        mb_source=train_source,
        model_inputs_to_streams=input_map,
        mb_size=minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config=checkpoint_config,
        # test_config=test_config,
        cv_config=validation_config
    ).train()
    end = time.time()
    print('The Network took {} secs to train.'.format(end - start))
    print('Saving the model here {}.'.format(os.path.join(model_path, model_name)))
    # save the model
    trainer.model.save(os.path.join(model_path, model_name))

    if profiling:
        stop_profiler()

# Train and evaluate the network.
def run(train_data, test_data, validation_data, minibatch_size=200, epoch_size=50000, num_quantization_bits=32,
                            warm_up=0, num_epochs=100, restore=True, log_to_file='logs.txt',
                            num_mbs_per_log=100, profiling=True):
    _cntk_py.set_computation_network_trace_level(0)

    network = create_resnet_network()

    progress_writers = [C.logging.ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=C.train.distributed.Communicator.rank(),
        num_epochs=num_epochs,
        distributed_freq=None)]

    trainer = create_trainer(network, epoch_size, num_quantization_bits, warm_up, progress_writers)
    train_source = create_image_mb_source(train_data, train=True, total_number_of_samples=num_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, train=False, total_number_of_samples=C.io.FULL_DATA_SWEEP)
    validation_source = create_image_mb_source(validation_data, train=False, total_number_of_samples=C.io.FULL_DATA_SWEEP)

    train_model(network, trainer, train_source, validation_source, test_source, minibatch_size, epoch_size, restore, profiling)

# Predict Images
def predict_image(model, image_path):
    # load and format image (resize, RGB -> BGR, CHW -> HWC)
    try:
        img = Image.open(image_path)
        resized = img.resize((image_width, image_height), Image.ANTIALIAS)
        bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
        hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

        # compute model output
        arguments = {model.arguments[0]: [hwc_format]}
        output = model.eval(arguments)

        # return softmax probabilities
        sm = C.softmax(output[0])
        return sm.eval()
    except Exception as e:
        print(e)
        print("Could not open (skipping file): {}".format(image_path))
        return None

# Evaluate the test set
def evaluate_model(model):
    train_dict = {}
    test_dict = {}
    confusion_matrix = {}
    cat_confusion_matrix = {}

    for label in os.listdir(train_dir):
        cat = label.split('-')[0]
        print(cat)

        print('Processing {}'.format(label))
        train_dict[label] = {
            'top_1': 0,
            'top_5': 0,
            'total': 0
        }

        id = label2id[label]

        for image in os.listdir(os.path.join(train_dir, label)):
            prediction = predict_image(model, os.path.join(train_dir, label, image))
            if prediction is not None:
                train_dict[label]['total'] += 1
                predicted_label = id2label[np.argmax(prediction)]
                if id == np.argmax(prediction):
                    train_dict[label]['top_1'] += 1
                if id in np.argsort(prediction)[-5:]:
                    train_dict[label]['top_5'] += 1

    for label in os.listdir(test_dir):
        cat = label.split('-')[0]
        print(cat)

        print('Processing {}'.format(label))
        test_dict[label] = {
            'top_1': 0,
            'top_5': 0,
            'total': 0
        }

        cat_confusion_matrix[cat] = {}
        confusion_matrix[label] = {}

        for other_label in os.listdir(test_dir):
            other_cat = other_label.split('-')[0]
            cat_confusion_matrix[cat][other_cat] = 0
            confusion_matrix[label][other_label] = 0

        id = label2id[label]

        for image in os.listdir(os.path.join(test_dir, label)):
            prediction = predict_image(model, os.path.join(test_dir, label, image))
            if prediction is not None:
                test_dict[label]['total'] += 1
                predicted_label = id2label[np.argmax(prediction)]
                pred_cat = predicted_label.split('-')[0]
                cat_confusion_matrix[cat][pred_cat] += 1
                confusion_matrix[label][predicted_label] += 1
                if id == np.argmax(prediction):
                    test_dict[label]['top_1'] += 1
                if id in np.argsort(prediction)[-5:]:
                    test_dict[label]['top_5'] += 1

    ccfm = pd.DataFrame(cat_confusion_matrix)
    cfm = pd.DataFrame(confusion_matrix)
    # with open('confusion-matrix', 'wb') as f:
    #     pickle.dump(confusion_matrix, f)

    train_df = pd.DataFrame(train_dict).transpose()
    train_df['top_1_%'] = (train_df['top_1'] / train_df['total']) * 100
    train_df['top_5_%'] = (train_df['top_5'] / train_df['total']) * 100

    test_df = pd.DataFrame(test_dict).transpose()
    test_df['top_1_%'] = (test_df['top_1'] / test_df['total']) * 100
    test_df['top_5_%'] = (test_df['top_5'] / test_df['total']) * 100

    writer = pd.ExcelWriter('evaluation-matrix.xlsx')
    train_df.to_excel(writer, 'Train')
    test_df.to_excel(writer, 'Test')
    ccfm.to_excel(writer, 'cat_confusion_matrix')
    cfm.to_excel(writer, 'confusion-matrix')
    writer.save()
    print('\nDone.\n')


if __name__=='__main__':
    train_data = os.path.join(data_path, 'train_map.txt')
    test_data = os.path.join(data_path, 'test_map.txt')
    validation_data = os.path.join(data_path, 'validation_map.txt')

    args = parser.parse_args()
    if args.train:
        print('\nStarting Training...\n')
        run(train_data, test_data, validation_data, epoch_size=epoch_size, num_epochs=50)

    if args.evaluate:
        print('\nStarting Evaluation...\n')
        model = C.load_model(args.model)
        evaluate_model(model)

    # Must call MPI finalize when process exit without exceptions
    C.train.distributed.Communicator.finalize()
