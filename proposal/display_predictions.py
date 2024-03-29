import os
import classifier
from utils.visualization_util import *
import sklearn.preprocessing
import parameters as params
import configuration as cfg

def run_demo():

    video_name = os.path.basename(cfg.sample_video_path).split('.')[0]

    # read video
    video_clips, num_frames = get_video_clips(cfg.sample_video_path)

    print("Number of clips in the video : ", len(video_clips))

    # build models
    original_model = keras.models.load_model(cfg.extractor_model_weights)
    feature_extractor = keras.models.Model(
        inputs = original_model.input,
        outputs = original_model.get_layer("lstm_1").output
    )
    classifier_model = build_classifier_model()

    print("Models initialized")

    # extract features
    rgb_features = []
    for i, clip in enumerate(video_clips):
        clip = np.array(clip)
        if len(clip) < params.frame_count:
            continue

        clip = preprocess_input(clip)
        rgb_feature = feature_extractor.predict(clip)[0]
        rgb_features.append(rgb_feature)

        print("Processed clip : ", i)

    rgb_features = np.array(rgb_features)
    rgb_feature_bag = interpolate(rgb_features, params.features_per_bag)
    
    # classify using the trained classifier model
    predictions = classifier_model.predict(rgb_feature_bag)

    predictions = np.array(predictions).squeeze()

    predictions = extrapolate(predictions, num_frames)
    
    save_path = os.path.join(cfg.output_folder, video_name + '.gif')
    # visualize predictions
    print('Executed Successfully - '+video_name + '.gif saved')
    visualize_predictions(cfg.sample_video_path, predictions, save_path)


if __name__ == '__main__':
    run_demo()
