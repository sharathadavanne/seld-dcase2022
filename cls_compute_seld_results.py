import os
import SELD_evaluation_metrics
import cls_feature_class
import parameters
import numpy as np


class ComputeSELDResults(object):
    def __init__(
            self, params, ref_files_folder=None, use_polar_format=True
    ):
        self._use_polar_format = use_polar_format
        self._desc_dir = ref_files_folder if ref_files_folder is not None else os.path.join(params['dataset_dir'], 'metadata_dev')
        self._doa_thresh = params['lad_doa_thresh']

        # Load feature class
        self._feat_cls = cls_feature_class.FeatureClass(params)
        
        # collect reference files
        self._ref_labels = {}
        for split in os.listdir(self._desc_dir):      
            for ref_file in os.listdir(os.path.join(self._desc_dir, split)):
                # Load reference description file
                gt_dict = self._feat_cls.load_output_format_file(os.path.join(self._desc_dir, split, ref_file))
                if not self._use_polar_format:
                    gt_dict = self._feat_cls.convert_output_format_polar_to_cartesian(gt_dict)
                nb_ref_frames = max(list(gt_dict.keys()))
                self._ref_labels[ref_file] = [self._feat_cls.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = params['average']

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_files_path, pred_file))
            if self._use_polar_format:
                pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
            pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])

            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])

        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        return ER, F, LE, LR, seld_scr, classwise_results

    def get_consolidated_SELD_results(self, pred_files_path, score_type_list=['all', 'room']):
        '''
            Get all categories of results.
            ;score_type_list: Supported
                'all' - all the predicted files
                'room' - for individual rooms

        '''

        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        nb_pred_files = len(pred_files)

        # Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)

        print('Number of predicted files: {}\nNumber of reference files: {}'.format(nb_pred_files, self._nb_ref_files))
        print('\nCalculating {} scores for {}'.format(score_type_list, os.path.basename(pred_output_format_files)))

        for score_type in score_type_list:
            print('\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------  {}   ---------------------------------------------'.format('Total score' if score_type=='all' else 'score per {}'.format(score_type)))
            print('---------------------------------------------------------------------------------------------------')

            split_cnt_dict = self.get_nb_files(pred_files, tag=score_type) # collect files corresponding to score_type
            # Calculate scores across files for a given score_type
            for split_key in np.sort(list(split_cnt_dict)):
                # Load evaluation metric class
                eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh, average=self._average)
                for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
                    # Load predicted output format file
                    pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_output_format_files, pred_file))
                    if self._use_polar_format:
                        pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
                    pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])

                    # Calculated scores
                    eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])

                # Overall SED and DOA scores
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

                print('\nAverage score for {} {} data using {} coordinates'.format(score_type, 'fold' if score_type=='all' else split_key, 'Polar' if self._use_polar_format else 'Cartesian' ))
                print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
                print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(ER, 100*F))
                print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(LE, 100*LR))

def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


if __name__ == "__main__":
    pred_output_format_files = 'results/1_1_dev_split0_accdoa_foa_20220428133518_val' # Path of the DCASEoutput format files
    params = parameters.get_params()
    # Compute just the DCASE final results 
    score_obj = ComputeSELDResults(params)
    ER, F, LE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(pred_output_format_files)
    print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
    print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(ER, 100*F))
    print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(LE, 100*LR))
    if params['average']=='macro':
        print('Classwise results on unseen test data')
        print('Class\tER\tF\tLE\tLR\tSELD_score')
        for cls_cnt in range(params['unique_classes']):
            print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(cls_cnt, classwise_test_scr[0][cls_cnt], classwise_test_scr[1][cls_cnt], classwise_test_scr[2][cls_cnt], classwise_test_scr[3][cls_cnt], classwise_test_scr[4][cls_cnt]))


    # UNCOMMENT to Compute DCASE results along with room-wise performance
    # score_obj.get_consolidated_SELD_results(pred_output_format_files)

