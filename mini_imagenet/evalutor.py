import torch

class ClassificationEvalutor:
    def __init__(self):
        self.correct = 0
        self.count = 0

    def add(self, outputs, targets):
        self.count += outputs.shape[0]
        preds = torch.argmax(outputs, dim=1)
        
        # Compare the two tensors element-wise
        is_correct = preds == targets
        num_correct = is_correct.sum().item()
        self.correct += num_correct
        return num_correct / outputs.shape[0]
            
    def summerize(self, ):
        return float(self.correct) / self.count 