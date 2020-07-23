# encoding: utf-8

import os.path as osp
from .bases import BaseImageDataset


class target_test(BaseImageDataset):
    """
    target_training: only constains camera ID, no class ID information

    Dataset statistics:

    """
    dataset_dir = 'target_test'

    def __init__(self, root='./example/data/challenge_datasets', verbose=True, **kwargs):
        super(target_test, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'image_query/')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_gallery/')
        self._check_before_run()
        self.query = self._process_dir(self.query_dir, 'index_test_query.txt')
        self.gallery = self._process_dir(self.gallery_dir, 'index_test_gallery.txt')
        if verbose:
            print("=> target_validation loaded")
            self.print_dataset_statistics_validation(self.query, self.gallery)
        

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, images_doc):
        image_list = osp.join(self.dataset_dir, images_doc)
        info = open(image_list).readlines() # image_name, image_id

        dataset = []
        for i in range(len(info)):
            element = info[i]
            image_name, image_id = element.split(' ')[0], element.split(' ')[1]
            pid = 0 # target test has no lables, set ID 0 for all images
            dataset.append((osp.join(dir_path, image_name), pid, image_id))
        return dataset
