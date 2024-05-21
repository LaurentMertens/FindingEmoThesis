"""

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os


class Config:
    HOME_DIR = os.path.expanduser('~')
    DIR_PARENT = os.path.join(HOME_DIR, 'Work', 'Projects', 'NeuroANN')
    PROJECT_ROOT = os.path.join(DIR_PARENT, 'Code', 'FindingEmoThesis')
    DIR_OUTPUT = os.path.join(PROJECT_ROOT, 'output')
    FILE_OPENIMAGES_CLASSES = os.path.join(PROJECT_ROOT, 'emonet_py', 'openimages.names')

    DIR_DATA = os.path.join(DIR_PARENT, 'Data')
    FILE_ANN3_SINGLE = os.path.join(DIR_DATA, 'FindingEmo Paper', 'annotations_single.ann')
