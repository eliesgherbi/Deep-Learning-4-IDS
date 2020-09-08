UNIVARIATE_DATASET_NAMES = []

"""
MTS_DATASET_NAMES = ['ArabicDigits', 'AUSLAN', 'CharacterTrajectories', 'CMUsubject16', 'ECG',
				'JapaneseVowels', 'KickvsPunch', 'Libras', 'NetFlow', 'UWave', 'Wafer', 'WalkvsRun']



[
					 
					 'unsup']

					 ['50x100_nofrq_playback']
					 
					 
					 '50x100_allATK',
 					 '50x100_continuous'
					 '50x100_plateau',
 					 '50x100_suppress',
 					 '50x100_flooding',
 					 '50x100_playback',
					 '50x100_nofrq_allATK',
 					 '50x100_nofrq_continuous',
 					 '50x100_nofrq_plateau',
 					 '50x100_nofrq_suppress',
 					 '50x100_nofrq_flooding',

"""

MTS_DATASET_NAMES = ['50x100_nofrq_plateau',
 					 '50x100_nofrq_suppress',
 					 '50x100_nofrq_flooding']
	
ITERATIONS = 1 # nb of random runs for random initializations

ARCHIVE_NAMES = ['mts_archive']#'UCR_TS_Archive_2015',

CLASSIFIERS = ['lstm']#,'tcn']#,'mlp','resnet','tlenet','mcnn','twiesn','encoder','mcdcnn','cnn']

dataset_types = {}

themes_colors ={'communication':'CAN'}
