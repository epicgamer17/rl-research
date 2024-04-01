class LinearAnnealedLRSchedule:
    def __init__(self, start_lr, end_lr, total_steps):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = total_steps
        self.current_lr = start_lr

    def get_lr(self, training_step):
        self.current_lr = self.start_lr - (
            (self.start_lr - self.end_lr) * training_step / self.total_steps
        )
        return self.current_lr
