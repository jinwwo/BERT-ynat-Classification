from transformers import TrainerCallback
import matplotlib.pyplot as plt
import sys

class LossCallback(TrainerCallback):
    def __init__(self, logging_interval=10):
        self.train_losses = []
        self.eval_losses = []
        self.count = 0
        self.logging_interval = logging_interval
        self.loss_tracker = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.loss_tracker.append(logs['loss'])
                
        if 'eval_loss' in logs:
            self.train_losses.append(sum(self.loss_tracker) / (len(self.loss_tracker) + sys.float_info.epsilon))
            self.eval_losses.append(logs['eval_loss'])
            self.loss_tracker = []
            
    def plot_loss(self, output='loss_plot.jpg'):
        print(f"self.train_losses: {self.train_losses}")
        print(f"self.eval_losses: {self.eval_losses}")
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.eval_losses) + 1)
        plt.plot(
            epochs, 
            self.train_losses, 
            label='Training Loss'
        )
        if any(self.eval_losses):
            plt.plot(
                epochs,
                self.eval_losses,
                label='Validation Loss'
            )
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig(output)
        plt.close()
        
    
            