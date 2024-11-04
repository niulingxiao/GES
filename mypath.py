class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset.lower() == 'defacto12k':
            return './data/defacto12k/'  # folder that contains VOCdevkit/.
        elif dataset.lower() == 'casiav2':
            return './data/Casiav2/'  # folder that contains dataset/.
        elif dataset.lower() == 'synthetic':
            return '/home/zzt/pan1/DATASETS/path_txt_files/'
        elif dataset.lower() == 'catnet':
            return './data/catnet/'
        # elif dataset == 'cityscapes':
        #     return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        # elif dataset == 'coco':
        #     return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
