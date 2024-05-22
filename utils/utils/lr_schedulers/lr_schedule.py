class LRScheduler:
    def __init__(self, lr_schedule_dict):
        self.lr_schedule_dict = lr_schedule_dict
        assert (
            0 in self.lr_schedule_dict.keys()
        ), "The first key in the lr_schedule_dict should be 0"
        self.current_lr = self.lr_schedule_dict[0]

    def get_lr(self, training_step):
        if training_step in self.lr_schedule_dict.keys():
            self.current_lr = self.lr_schedule_dict[training_step]
            return self.current_lr
        else:
            return self.current_lr
