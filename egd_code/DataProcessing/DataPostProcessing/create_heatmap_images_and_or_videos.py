import pandas as pd
import multiprocessing
import sys
from scipy import ndimage
import cv2
import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skvideo.io import FFmpegWriter
import glob

matplotlib.use("Agg")

# Modified from original script to use JPEGs instead of DICOMs
# Replace with the location of the MIMIC-CXR images
original_folder_images = '/media/hdd/mimic-cxr-jpg'
egd_path = '/media/hdd/mimic-cxr-eye/'


def create_videos(input_folder, eye_gaze_table, cases_table, data_type):
    """
    This method is optional. It just creates videos of heatmaps using heatmap frames for particular eye_gaze_table.
    It can ONLY run after process_eye_gaze_table() method finishes.

    :param input_folder: Folder with saved heatmap frames (see process_eye_gaze_table())
    :param eye_gaze_table: Pandas dataframe containing the eye gaze data
    :param cases_table Pandas dataframe containing case information (master_table.csv)
    :param data_type: Type of eye gaze type: fixations, raw eye gaze
    :return: None
    """

    try:
        os.mkdir(input_folder)
    except IOError:
        pass

    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    for subfolder in subfolders:
        print('Subfolder\n',subfolder.split('/')[-1])
        files = glob.glob(os.path.join(subfolder,"*frame.png"))

        # Extract image name
        image_name = subfolder.split('/')[-1]

        # Get file path to original image
        case_index = cases_table.loc[cases_table['dicom_id'] == image_name].index[0]
        file_path = cases_table.loc[case_index, 'path']
        file_path = file_path[:-4] + '.jpg'

        try:
            os.mkdir(os.path.join(input_folder,image_name))
        except IOError:
            pass

        for i in range(len(files)):
            if i == 0:
                try:
                    # Load CXR image
                    image = cv2.imread(os.path.join(original_folder_images, file_path)).astype('uint8')

                    # Resize it to be 1080x1920
                    image = cv2.resize(image, (1920, 1080))
                except IOError:
                    # else it is a calibration image
                    image = cv2.imread('calibration_image.png').astype('uint8')

                try:
                    last_row = eye_gaze_table.index[eye_gaze_table['DICOM_ID'] == image_name].tolist()[-1]
                    full_time = eye_gaze_table["Time (in secs)"].values[last_row]
                    fps = (len(files)) / full_time
                except ZeroDivisionError:
                    print('Error with fps!')

                if (image.shape[0] % 2) > 0:
                    image = np.vstack((image, np.zeros((1, image.shape[1], 3))))
                if (image.shape[1] % 2) > 0:
                    image = np.hstack((image, np.zeros((image.shape[0], 1, 3))))

                crf = 23
                vid_out = FFmpegWriter(os.path.join(input_folder, image_name, data_type+'.mp4'),
                                       inputdict={'-r': str(fps),
                                                  '-s': '{}x{}'.format(image.shape[1], image.shape[0])},
                                       outputdict={'-r': str(fps), '-c:v': 'mpeg4', '-crf': str(crf),
                                                   '-preset': 'ultrafast',
                                                   '-pix_fmt': 'yuv420p'}, verbosity=0
                                       )

            try:
                overlay_heatmap = cv2.addWeighted(image.astype('uint8'), 0.5, cv2.imread(os.path.join(subfolder,str(i+1)+'_frame.png')).astype('uint8'), 0.5, 0)
            except BaseException:
                print('error ', cv2.imread(files[i]).astype('uint8').shape)

            vid_out.writeFrame(overlay_heatmap)
        vid_out.close()


def calibrate(eye_gaze_table, screen_width=1920, screen_height=1080):
    """
    This method uses calibration image (read paper for more) and recalibrates coordinates

    :param gaze_table: pandas dataframe with eye gaze table
    :param screen_width: DO NOT CHANGE. This was used in the original eye gaze experiment.
    :param screen_height: DO NOT CHANGE. This was used in the original eye gaze experiment.
    :return:
    """
    try:
        calibrationX=[]
        calibrationY=[]

        # Iterate through each image in the raw eye gaze spreadsheet
        for index, row in eye_gaze_table.iterrows():
            image_name = row['DICOM_ID']

            if not os.path.exists(os.path.join(original_folder_images, image_name + '.jpg')):
                last_row = eye_gaze_table.index[eye_gaze_table['DICOM_ID'] == image_name ].tolist()[-1]
                eyeX = eye_gaze_table['FPOGX'][last_row] * screen_width
                eyeY = eye_gaze_table['FPOGY'][last_row] * screen_height

                # Get pixel coordinates from raw eye gaze coordinates
                # eyeX = row['FPOGX'] * screen_width
                # eyeY = row['FPOGY'] * screen_height
                calibrationX.append(eyeX)
                calibrationY.append(eyeY)

        calibrationX = np.asarray(calibrationX)
        calibrationY = np.asarray(calibrationY)

        mean_X = np.mean(calibrationX)
        mean_Y = np.mean(calibrationY)

        calibratedX = screen_width//2 - mean_X
        calibratedY = screen_height//2 - mean_Y

        return calibratedX, calibratedY
    except BaseException:
        print('No calibration available')
        return .0, .0


def process_eye_gaze_table(session_table, export_folder, cases, window=0, calibration=False, sigma=150,
                           screen_width=1920, screen_height=1080):
    """
    Main method to process eye gaze session table (e.g. fixations or raw eye gaze) to create heatmap frames for each
    coordinate.
    The frames are saved in export_folder/dicom_id
    It returns the same session table with:
        a) its eye gaze coordinates (i.e. FPOGX, FPOGY) mapped to image coordinates (i.e. X_ORIGINAL, Y_ORIGINAL).
        b) each row's heatmap frame (i.e. EYE_GAZE_HEATMAP_FRAME)

    This method also allows the user to do the following too:

    - Re-calibrate coordinates (i.e. FPOGX, FPOGY) by utilizing the calibration template (i.e. calibration_image.png)
    if available in this particular session
    - Use exponential decay as a weight in a specific window (i.e. +- heatmap frames on a given heatmap frame) for the
    given heatmap frame
    - Apply different sigma size when generating heatmap frames

    :param session_table: a fixation or raw eye gaze Pandas dataframe for a particular session
    :param export_folder: folder to save heatmap frames
    :param cases: the original master sheet in Pandas dataframe
    :param window: number of frames to use when applying exponential decay on a given heatmap fram
    :param calibration: flag to perform re-calibration
    :param original_folder_images: location of original dicom images downloaded from MIMIC source
    :param sigma: sigma of gaussian to apply on a given eye gaze point
    :param screen_width: screen width in pixels for monitor's resolution used in experiment. DO NOT CHANGE!
    :param screen_height: screen height in pixels for monitor's resolution used in experiment. DO NOT CHANGE!
    :return: processed session table with eye gaze coordinated mapped to original image coordinates
    """

    session_table["X_ORIGINAL"] = ""
    session_table["Y_ORIGINAL"] = ""
    session_table["EYE_GAZE_HEATMAP_FRAME"] = ""
    previous_image_name = ''

    heatmaps = []
    counter = 1

    # Do calibration
    if calibration:
        calibratedX, calibratedY = calibrate(session_table)
    else:
        calibratedX = calibratedY = .0

    # Iterate through each image in the raw eye gaze spreadsheet
    for index, row in session_table.iterrows():
        # Get pixel coordinates from raw eye gaze coordinates and calibrate them
        eyeX = row['FPOGX'] * screen_width + calibratedX
        eyeY = row['FPOGY'] * screen_height + calibratedY

        # Get image name
        # print(row)
        image_name = row['DICOM_ID']

        # Get file path
        try:
            case_index = cases.loc[cases['dicom_id'] == image_name].index[0]
            file_path = cases.iloc[case_index]['path']
            file_path = file_path[:-4] + '.jpg'
        except IndexError:
            # For some reason we can't find the dicom id...
            continue

        # print(image_name, index, session_table.shape[0], previous_image_name, image_name)
        # Condition to start a new eye gaze drawing job
        if previous_image_name != image_name:
            counter = 1
            if previous_image_name != '':
                print('Finished ', index, '/', session_table.shape[0], ' rows from session ',
                      session_table['SESSION_ID'].values[0])

                for i in range(len(heatmaps)):
                    if window != 0:
                        left_window = right_window=window
                        if i - window < 0:
                            left_window = i
                        if i + window > len(heatmaps):
                            right_window = len(heatmaps) - i

                        for j in range(i - left_window, i + right_window):
                            # Use exponential decay relative to length of existing observed eye gaze
                            decay = math.exp(-abs(i - j))
                            heatmaps[j] *= decay

                        heatmap_numpy = heatmaps[i - left_window:i + right_window]
                        current_heatmap = np.sum(heatmap_numpy, axis=0)
                    else:
                        current_heatmap = heatmaps[i]

                    plt.imsave(os.path.join(export_folder, previous_image_name, str(i) + '_frame.png'),
                               ndimage.gaussian_filter(current_heatmap, sigma))

                heatmap = ndimage.gaussian_filter(record, sigma)

                try:
                    os.mkdir(os.path.join(export_folder, previous_image_name))
                except IOError:
                    pass

                plt.imsave(os.path.join(export_folder, previous_image_name, 'heatmap.png'), heatmap)
                heatmaps = []
                del current_heatmap

            if not os.path.exists(os.path.join(export_folder, image_name)):
                os.mkdir(os.path.join(export_folder, image_name))

            if os.path.exists(os.path.join(original_folder_images, file_path)):
                image = cv2.imread(os.path.join(original_folder_images, file_path))

                try:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                except cv2.error:
                    # Our image is probably already RGB...
                    pass
                # Load metadata (top, bottom, left, right padding pixel dimensions) about the particular jpeg image
                # from the master spreadsheet
                case_index = cases.loc[cases['dicom_id'] == image_name].index[0]
                top, bottom, left, right = cases.iloc[case_index]['image_top_pad'], \
                                           cases.iloc[case_index]['image_bottom_pad'],\
                                           cases.iloc[case_index]['image_left_pad'],\
                                           cases.iloc[case_index]['image_right_pad']
            else:
                image = np.zeros((screen_height, screen_width,3), dtype=np.uint8)
                top, bottom, left, right = (0, 0, 0, 0)

            if (image.shape[0] % 2) > 0:
                image = np.vstack((image, np.zeros((1, image.shape[1], 3))))
            if (image.shape[1] % 2) > 0:
                image = np.hstack((image, np.zeros((image.shape[0], 1, 3))))

            record = np.zeros([image.shape[0], image.shape[1]])
            previous_image_name = image_name

        try:
            # Keep eye gazes that fall within the image
            if eyeX > left and eyeX < screen_width-right and eyeY> top and eyeY < screen_height-bottom:
                x_original = eyeX - left
                y_original = eyeY - top
            else:
                x_original = -1
                y_original = -1

            # Remap to original image coordinates
            resized_width = screen_width - left - right
            resized_height = screen_height - top - bottom
            x_original_in_pixels = int((image.shape[1] / resized_width) * x_original)
            y_original_in_pixels = int((image.shape[0] / resized_height) * y_original)

            # Create heatmap
            heatmap_image = np.zeros([image.shape[0], image.shape[1]])
            if y_original_in_pixels>0:
                record[int(y_original_in_pixels), int(x_original_in_pixels)] += 1
                heatmap_image[int(y_original_in_pixels), int(x_original_in_pixels)] = 1
            heatmaps.append(heatmap_image)

            # Also save eye gazes coordinates to the spreadsheet
            session_table.loc[index,"X_ORIGINAL"]=x_original_in_pixels
            session_table.loc[index,"Y_ORIGINAL"]=y_original_in_pixels
            session_table.loc[index,"EYE_GAZE_HEATMAP_FRAME"] = str(counter) + '_frame.png'
            counter += 1
        except BaseException:
            print(sys.exc_info()[0])

    return session_table


def concatenate_session_tables(eye_gaze_session_tables):
    """
    Auxiliary method that simply concatenates each individual session eye gaze table into a single table
    :param eye_gaze_session_tables: List of Pandas dataframes of session eye gaze tables
    :return:
    """
    final_table = []

    for i, table in enumerate(eye_gaze_session_tables):
        if i == 0:
            n_columns = len(table.columns)
            columns = table.columns
            table.columns = range(n_columns)
            final_table = table
        else:
            table.columns = range(n_columns)
            final_table = pd.concat([final_table, table], axis=0, ignore_index=True, sort=False)

    final_table.columns = columns
    return final_table


def process_fixations(experiment_name, video=False):

    print('--------> FIXATIONS <--------')

    cases = pd.read_csv(os.path.join(egd_path, 'master_sheet.csv'))
    table = pd.read_csv(os.path.join(egd_path, 'fixations.csv'))

    sessions = table.groupby(['SESSION_ID'])

    try:
        os.makedirs(experiment_name, exist_ok=True)
    except OSError as exc:
        print(exc, ' Proceeding...')
        pass

    p = multiprocessing.Pool(processes=len(sessions))
    objects = []
    for session in sessions:
        df = session[1].copy().reset_index(drop=True)
        objects.append((df, experiment_name, cases))
    eye_gaze_session_tables = p.starmap(process_eye_gaze_table, [i for i in objects])
    p.close()

    final_table = concatenate_session_tables(eye_gaze_session_tables)

    # Save experiment consolidated table
    final_table.to_csv(experiment_name + '.csv', index=False)

    # Create video files with fixation heatmaps
    if video:
        create_videos(experiment_name, final_table, cases, data_type='fixations')

def process_raw_eye_gaze(experiment_name, video=False):

    print('--------> RAW EYE GAZE <--------')

    cases = pd.read_csv(os.path.join(egd_path, 'master_sheet.csv'))
    table = pd.read_csv(os.path.join(egd_path, 'fixations.csv'))

    sessions = table.groupby(['SESSION_ID'])

    try:
        os.mkdir(experiment_name)
    except IOError:
        pass

    p = multiprocessing.Pool(processes=len(sessions))
    objects = []
    for session in sessions:
        df = session[1].copy().reset_index(drop=True)
        objects.append((df, experiment_name, cases))
    eye_gaze_session_tables = p.starmap(process_eye_gaze_table, [i for i in objects])
    p.close()

    final_table = concatenate_session_tables(eye_gaze_session_tables)

    # Save experiment consolidated table
    final_table.to_csv(experiment_name + '.csv', index=False)

    # Create video files with raw eye gaze heatmaps
    if video:
        create_videos(experiment_name, final_table, cases, data_type='raw_eye_gaze')


if __name__ == '__main__':
    # FOR fixations.csv: To generate heatmap images and create videos of the heatmaps, uncomment the following line
    #process_fixations(experiment_name='/media/hdd/mimic-cxr-eye/fixation-heatmaps', video=True)

    eye_gaze_table = pd.read_csv(os.path.join(egd_path, 'fixations.csv'))
    master_table = pd.read_csv(os.path.join(egd_path, 'master_sheet.csv'))
    create_videos('/media/hdd/mimic-cxr-eye/fixation-heatmaps/', eye_gaze_table, master_table, 'fixations')

    #The following method is required only if you want to work with the raw eye gaze data
    # (as they come from the machine unprocessed). Please read paper for the differences.
    # process_raw_eye_gaze(experiment_name='eye_gaze_heatmaps')
