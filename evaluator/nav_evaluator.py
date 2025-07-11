import numpy as np
from fastdtw import fastdtw


'''
result: {
    episode_id,
    gt_traj,
    pred_traj,
    
'''
class LmNavEvaluator:
    def __init__(self):
        self.metrics = {
            "sr": 0.0,
            "osr": 0.0,
            "spl": 0.0,
            "sdtw":0.0,
            "ne": 0.0
        }
        self.succ_thresh = 20.0
        self.current_status = {
            "gt_traj": [],
            "pred_traj": [],
            "ep_success": 0.0,
            "gt_traj_len": 0.0,
            "pred_traj_len": 0.0
        }

        self.sr_scorer = []
        self.osr_scorer = []
        self.spl_scorer = []
        # self.ndtw_scorer = []
        self.sdtw_scorer = []
        self.ne_scorer = []

    def update(self, output):
        """
        update result of a single traj
        :param output: result of a single trajectory
        :return:
        """
        self.status_updator(output)

        self.sr_updator()
        self.osr_updator()
        self.ne_updator()
        self.sdtw_updator()
        self.spl_updator()

    def calculate_metrics(self):
        print(self.sr_scorer)
        print(self.osr_scorer)
        print(self.ne_scorer)
        print(self.spl_scorer)
        print(self.sdtw_scorer)

        sr = np.sum(np.array(self.sr_scorer))*1.0 / len(self.sr_scorer)
        osr = np.sum(np.array(self.osr_scorer))*1.0/len(self.osr_scorer)
        ne = np.sum(np.array(self.ne_scorer))*1.0/len(self.ne_scorer)
        spl = np.sum(np.array(self.spl_scorer))*1.0/len(self.spl_scorer)
        sdtw = np.sum(np.array(self.sdtw_scorer))*1.0/len(self.sdtw_scorer)

        self.metrics['sr'] = sr
        self.metrics['osr'] = osr
        self.metrics['ne'] = ne
        self.metrics['spl'] = spl
        self.metrics['sdtw'] = sdtw

    def log_metrics(self):
        self.calculate_metrics()
        print(f"sr: {self.metrics['sr']}, spl: {self.metrics['spl']}, osr: {self.metrics['osr']}, sdtw: {self.metrics['sdtw']}, ne: {self.metrics['ne']}")

    def status_updator(self, output):
        updated_status = output

        gt_traj = np.array(output['gt_traj'])
        pred_traj = np.array(output['pred_traj'])

        if np.linalg.norm(gt_traj[-1]-pred_traj[-1]) < self.succ_thresh:
            updated_status['ep_success'] = 1.0
        else:
            updated_status['ep_success'] = 0.0

        gt_traj_diff = gt_traj[1:] - gt_traj[:-1]
        gt_traj_len = np.sum(np.linalg.norm(gt_traj_diff, axis=1))

        pred_traj_diff = pred_traj[1:] - pred_traj[:-1]
        pred_traj_len = np.sum(np.linalg.norm(pred_traj_diff, axis=1))

        updated_status['gt_traj_len'] = gt_traj_len
        updated_status['pred_traj_len'] = pred_traj_len

        self.current_status = updated_status

    def sr_updator(self):
        self.sr_scorer.append(self.current_status['ep_success'])

    def osr_updator(self):
        flag = 0
        gt_traj = np.array(self.current_status['gt_traj'])
        pred_traj = np.array(self.current_status['pred_traj'])

        for i in range(len(pred_traj)):
            loc = np.linalg.norm(pred_traj[i]-gt_traj[-1])
            if loc < self.succ_thresh:
                flag = 1
                break
        self.osr_scorer.append(flag)

    def ne_updator(self):
        gt_traj = np.array(self.current_status['gt_traj'])
        pred_traj = np.array(self.current_status['pred_traj'])

        ne = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
        self.ne_scorer.append(ne)

    def sdtw_updator(self):
        gt_traj = np.array(self.current_status['gt_traj'])
        pred_traj = np.array(self.current_status['pred_traj'])

        dtw_dist = fastdtw(gt_traj, pred_traj)[0]
        nDTW = np.exp(-dtw_dist / (len(gt_traj) * self.succ_thresh))

        sDTW = self.current_status['ep_success'] * nDTW
        self.sdtw_scorer.append(sDTW)

    def spl_updator(self):
        ep_succ = self.current_status['ep_success']
        gt_traj_len = self.current_status['gt_traj_len']
        pred_traj_len = self.current_status['pred_traj_len']

        ep_spl = ep_succ * gt_traj_len / (max(gt_traj_len, pred_traj_len)+1e-6)
        self.spl_scorer.append(ep_spl)


class CityNavEvaluator:
    def __init__(self):
        self.metrics = {
            "sr": 0.0,
            "osr": 0.0,
            "spl": 0.0,
            "sdtw":0.0,
            "ne": 0.0
        }
        self.succ_thresh = 20.0
        self.current_status = {
            "gt_traj": [],
            "pred_traj": [],
            "ep_success": 0.0,
            "gt_traj_len": 0.0,
            "pred_traj_len": 0.0
        }

        self.sr_scorer = []
        self.osr_scorer = []
        self.spl_scorer = []
        # self.ndtw_scorer = []
        self.sdtw_scorer = []
        self.ne_scorer = []

    def reset(self):
        self.metrics = {
            "sr": 0.0,
            "osr": 0.0,
            "spl": 0.0,
            "sdtw":0.0,
            "ne": 0.0
        }
        self.succ_thresh = 20.0
        self.current_status = {
            "gt_traj": [],
            "pred_traj": [],
            "ep_success": 0.0,
            "gt_traj_len": 0.0,
            "pred_traj_len": 0.0
        }

        self.sr_scorer = []
        self.osr_scorer = []
        self.spl_scorer = []
        self.sdtw_scorer = []
        self.ne_scorer = []

    def update(self, output):
        self.status_updator(output)

        self.sr_updator()
        self.osr_updator()
        self.ne_updator()
        self.sdtw_updator()
        self.spl_updator()

    def calculate_metrics(self):
        # print(self.sr_scorer)
        # print(self.osr_scorer)
        # print(self.ne_scorer)
        # print(self.spl_scorer)
        # print(self.sdtw_scorer)

        sr = np.sum(np.array(self.sr_scorer))*1.0 / len(self.sr_scorer)
        osr = np.sum(np.array(self.osr_scorer))*1.0/len(self.osr_scorer)
        ne = np.sum(np.array(self.ne_scorer))*1.0/len(self.ne_scorer)
        spl = np.sum(np.array(self.spl_scorer))*1.0/len(self.spl_scorer)
        sdtw = np.sum(np.array(self.sdtw_scorer))*1.0/len(self.sdtw_scorer)

        self.metrics['sr'] = sr
        self.metrics['osr'] = osr
        self.metrics['ne'] = ne
        self.metrics['spl'] = spl
        self.metrics['sdtw'] = sdtw

    def log_metrics(self):
        self.calculate_metrics()
        print(f"SR: {self.metrics['sr']}, SPL: {self.metrics['spl']}, OSR: {self.metrics['osr']}, SDTW: {self.metrics['sdtw']}, NE: {self.metrics['ne']}")

    def status_updator(self, output):
        updated_status = output

        gt_traj = np.array(output['gt_traj'])
        pred_traj = np.array(output['pred_traj'])

        if np.linalg.norm(gt_traj[-1]-pred_traj[-1]) < self.succ_thresh:
            updated_status['ep_success'] = 1.0
        else:
            updated_status['ep_success'] = 0.0

        gt_traj_diff = gt_traj[1:] - gt_traj[:-1]
        gt_traj_len = np.sum(np.linalg.norm(gt_traj_diff, axis=1))

        pred_traj_diff = pred_traj[1:] - pred_traj[:-1]
        pred_traj_len = np.sum(np.linalg.norm(pred_traj_diff, axis=1))

        updated_status['gt_traj_len'] = gt_traj_len
        updated_status['pred_traj_len'] = pred_traj_len

        self.current_status = updated_status

    def sr_updator(self):
        self.sr_scorer.append(self.current_status['ep_success'])

    def osr_updator(self):
        flag = 0
        gt_traj = np.array(self.current_status['gt_traj'])
        pred_traj = np.array(self.current_status['pred_traj'])

        for i in range(len(pred_traj)):
            loc = np.linalg.norm(pred_traj[i]-gt_traj[-1])
            if loc < self.succ_thresh:
                flag = 1
                break
        self.osr_scorer.append(flag)

    def ne_updator(self):
        gt_traj = np.array(self.current_status['gt_traj'])
        pred_traj = np.array(self.current_status['pred_traj'])

        ne = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
        self.ne_scorer.append(ne)

    def sdtw_updator(self):
        gt_traj = np.array(self.current_status['gt_traj'])
        pred_traj = np.array(self.current_status['pred_traj'])

        dtw_dist = fastdtw(gt_traj, pred_traj)[0]
        nDTW = np.exp(-dtw_dist / (len(gt_traj) * self.succ_thresh))

        sDTW = self.current_status['ep_success'] * nDTW
        self.sdtw_scorer.append(sDTW)

    def spl_updator(self):
        ep_succ = self.current_status['ep_success']
        gt_traj_len = self.current_status['gt_traj_len']
        pred_traj_len = self.current_status['pred_traj_len']

        ep_spl = ep_succ * gt_traj_len / (max(gt_traj_len, pred_traj_len)+1e-6)
        self.spl_scorer.append(ep_spl)
