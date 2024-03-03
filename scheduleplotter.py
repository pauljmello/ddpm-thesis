import torch
import matplotlib.pyplot as plt
import numpy as np


class scheduleplotter:
    def __init__(self, t_steps, device):
        self.t_steps = t_steps
        self.device = device

    def linear_schedule(self, t):
        return torch.linspace(1e-4, 2e-2, t)

    def cosine_schedule(self, t):
        x = torch.linspace(0, self.t_steps, t + 1, device=self.device)
        y = torch.cos(((x / self.t_steps) + 0.01) / (1 + 0.01) * torch.pi * 0.5) ** 2
        z = y / y[0]
        return torch.clip((1 - (z[1:] / z[:-1])), 0.0001, 0.9999).to(self.device)

    def sigmoid_schedule(self, t, start, end):
        sequence = torch.linspace(0, self.t_steps, t + 1, dtype=torch.float32, device=self.device) / self.t_steps
        v_start = torch.tensor(start, dtype=torch.float32, device=self.device).sigmoid()
        v_end = torch.tensor(end, dtype=torch.float32, device=self.device).sigmoid()
        alpha = (-((sequence * (end - start) + start) / 1).sigmoid() + v_end) / (v_end - v_start)
        alpha = alpha / alpha[0]
        betas = 1 - (alpha[1:] / alpha[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def plot_schedule(self, schedule_type, *args):
        if schedule_type == "linear":
            y = self.linear_schedule(self.t_steps)
            title = "Linear Schedule"
        elif schedule_type == "cosine":
            y = self.cosine_schedule(self.t_steps)
            title = "Cosine Schedule"
        elif schedule_type == "sigmoid":
            y = self.sigmoid_schedule(self.t_steps, *args)
            title = "Sigmoid Schedule"
        else:
            raise ValueError("Invalid schedule type")

        x = np.arange(1, self.t_steps + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y.cpu().numpy(), label=schedule_type)
        plt.title(title)
        plt.xlabel("Timesteps")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all_schedules(self, sigmoid_start=0, sigmoid_end=10, SNR_start=0.01, SNR_end=0.1):
        plt.figure(figsize=(12, 8))
        y_linear = self.linear_schedule(self.t_steps)
        plt.plot(np.arange(1, self.t_steps + 1), y_linear.cpu().numpy(), label="Linear")
        y_cosine = self.cosine_schedule(self.t_steps)
        plt.plot(np.arange(1, self.t_steps + 1), y_cosine.cpu().numpy(), label="Cosine")
        y_sigmoid = self.sigmoid_schedule(self.t_steps, sigmoid_start, sigmoid_end)
        plt.plot(np.arange(1, self.t_steps + 1), y_sigmoid.cpu().numpy(), label="Sigmoid")
        plt.title("Comparison of Different Schedules")
        plt.xlabel("Timesteps")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        plt.savefig("images/Comparison of Different Schedules.png")
class enhancedscheduleplotter:
    def __init__(self, t_steps, device):
        self.t_steps = t_steps
        self.device = device

    def linear_schedule(self, t, warmup_ratio=0.1):
        y = torch.linspace(1e-4, 2e-2, t)
        warmup_steps = int(t * warmup_ratio)
        y[:warmup_steps] = torch.linspace(1e-4, y[warmup_steps], warmup_steps)
        return y

    def cosine_schedule(self, t, warmup_ratio=0.1):
        x = torch.linspace(0, self.t_steps, t + 1, device=self.device)
        y = torch.cos(((x / self.t_steps) + 0.01) / (1 + 0.01) * torch.pi * 0.5) ** 2
        z = y / y[0]
        schedule = torch.clip((1 - (z[1:] / z[:-1])), 0.0001, 0.9999).to(self.device)
        warmup_steps = int(t * warmup_ratio)
        schedule[:warmup_steps] = torch.linspace(0.0001, schedule[warmup_steps], warmup_steps)
        return schedule

    def sigmoid_schedule(self, t, start, end, warmup_ratio=0.1):
        sequence = torch.linspace(0, self.t_steps, t + 1, dtype=torch.float32, device=self.device) / self.t_steps
        v_start = torch.tensor(start, dtype=torch.float32, device=self.device).sigmoid()
        v_end = torch.tensor(end, dtype=torch.float32, device=self.device).sigmoid()
        alpha = (-((sequence * (end - start) + start) / 1).sigmoid() + v_end) / (v_end - v_start)
        alpha = alpha / alpha[0]
        betas = 1 - (alpha[1:] / alpha[:-1])
        schedule = torch.clamp(betas, 0.0001, 0.9999)
        warmup_steps = int(t * warmup_ratio)
        schedule[:warmup_steps] = torch.linspace(0.0001, schedule[warmup_steps], warmup_steps)
        return schedule

    def hybrid_schedule(self, t, transition_point=0.5):
        first_half_steps = int(t * transition_point)
        second_half_steps = t - first_half_steps
        first_half = self.sigmoid_schedule(first_half_steps, 0, 10)
        second_half = self.linear_schedule(second_half_steps)
        return torch.cat((first_half, second_half[first_half_steps:]))

    def plot_schedule(self, schedule_type, *args):
        if schedule_type == "linear":
            y = self.linear_schedule(self.t_steps, *args)
            title = "Linear Schedule with Warmup"
        elif schedule_type == "cosine":
            y = self.cosine_schedule(self.t_steps, *args)
            title = "Cosine Schedule with Warmup"
        elif schedule_type == "sigmoid":
            y = self.sigmoid_schedule(self.t_steps, *args)
            title = "Sigmoid Schedule with Warmup"
        elif schedule_type == "hybrid":
            y = self.hybrid_schedule(self.t_steps, *args)
            title = "Hybrid Schedule (Sigmoid + Linear)"
        else:
            raise ValueError("Invalid schedule type")

        x = np.arange(1, self.t_steps + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y.cpu().numpy(), label=schedule_type)
        plt.title(title)
        plt.xlabel("Timesteps")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all_schedules(self, sigmoid_start=0, sigmoid_end=10, transition_point=0.5):
        plt.figure(figsize=(12, 8))

        # Linear schedule
        y_linear = self.linear_schedule(self.t_steps)
        plt.plot(np.arange(1, self.t_steps + 1), y_linear.cpu().numpy(), label="Linear")

        # Cosine schedule
        y_cosine = self.cosine_schedule(self.t_steps)
        plt.plot(np.arange(1, self.t_steps + 1), y_cosine.cpu().numpy(), label="Cosine")

        # Sigmoid schedule
        y_sigmoid = self.sigmoid_schedule(self.t_steps, sigmoid_start, sigmoid_end)
        plt.plot(np.arange(1, self.t_steps + 1), y_sigmoid.cpu().numpy(), label="Sigmoid")

        # Hybrid schedule
        y_hybrid = self.hybrid_schedule(self.t_steps, transition_point)
        plt.plot(np.arange(1, len(y_hybrid) + 1), y_hybrid.cpu().numpy(), label="Hybrid")

        plt.title("Comparison of Different Enhanced Schedules")
        plt.xlabel("Timesteps")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        plt.savefig("images/Comparison of Different Enhanced Schedules.png")


t_steps = 1000  # Number of timesteps
device = "cpu"  # Device can be "cpu" or "cuda" if using GPU

# Create an instance of the SchedulePlotter class and plot all schedules
plotter = scheduleplotter(t_steps, device)
plotter.plot_all_schedules(sigmoid_start=0, sigmoid_end=10, SNR_start=0.01, SNR_end=0.1)

# Create an instance of the EnhancedSchedulePlotter class and plot all schedules
enhanced_plotter = enhancedscheduleplotter(t_steps, device)
enhanced_plotter.plot_all_schedules(sigmoid_start=0, sigmoid_end=10, transition_point=.999)