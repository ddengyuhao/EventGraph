from abc import ABC, abstractmethod

class BaseMethod(ABC):
    def __init__(self, args, model):
        """
        Initialize the base method.

        Args:
            args: Arguments containing budget, hyperparameters, etc.
            model: The loaded backbone model (e.g., Video-LLaVA, Qwen-VL).
                   Note: Some methods (like FastV) may need to modify the model internals.
        """
        self.args = args
        self.model = model
        self.token_budget = args.token_budget
        # Temperature parameter for methods like Q-Frame
        self.temperature = getattr(args, 'temperature', 1.0)

    @abstractmethod
    def process_and_inference(self, video_path, question, options):
        """
        Core interface for processing and inference.

        Pipeline:
        1. Receive video path and user query.
        2. Execute method-specific logic (Compression / Selection / Memory).
        3. Call the model for inference.
        4. Return the prediction (Choice index or Text).

        Args:
            video_path (str): Path to the video file.
            question (str): The user query.
            options (list): List of candidate options (for multiple-choice tasks).

        Returns:
            str: The predicted answer.
        """
        pass