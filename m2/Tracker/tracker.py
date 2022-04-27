import json
from Tracker.bytetrack.bytetrack import ByteTrack

class MultiObjectTracker(object):
    def __init__(
        self,
        fps=30,
        use_gpu=False,
    ):
        self.fps = round(fps, 2)
        self.tracker = None
        self.config = None
        self.use_gpu = use_gpu

     

        self.use_gpu = False 

        with open('Tracker/bytetrack/config.json') as fp:
            self.config = json.load(fp)

        if self.config is not None:
            self.tracker = ByteTrack(
                fps=self.fps,
                track_thresh=self.config['track_thresh'],
                track_buffer=self.config['track_buffer'],
                match_thresh=self.config['match_thresh'],
                min_box_area=self.config['min_box_area'],
                mot20=self.config['mot20'],
            )

    def __call__(self, image, bboxes, scores, class_ids):
        if self.tracker is not None:
            results = self.tracker(
                image,
                bboxes,
                scores,
                class_ids,
            )
        else:
            raise ValueError('Tracker is None')

        # 0:Tracker ID, 1:Bounding Box, 2:Score, 3:Class ID
        return results[0], results[1], results[2], results[3]

    def print_info(self):
        from pprint import pprint

        print('Tracker')
        print('FPS:', self.fps)
        print('GPU:', self.use_gpu)
        pprint(self.config, indent=4)
        print()
