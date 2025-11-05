"""
Liquid Time-Constant Networks for NeuralSleep

Based on:
Hasani, R., et al. (2021). Liquid Time-constant Networks. AAAI.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Tuple, Optional


class LTCNeuron(nn.Module):
    """
    Liquid Time-Constant Neuron with adaptive dynamics

    Key Features:
    - Continuous-time dynamics via ODEs
    - Adaptive time constants
    - No discrete time steps
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_constant_range: Tuple[float, float] = (0.1, 10.0)
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_min, self.tau_max = time_constant_range

        # Learnable parameters
        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_rec = nn.Linear(hidden_size, hidden_size)

        # Time constants (learnable, initialized to log-space)
        self.tau_log = nn.Parameter(
            torch.randn(hidden_size) * 0.5 +
            torch.log(torch.tensor((self.tau_min + self.tau_max) / 2))
        )

        # Bias
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """
        Compute dhdt at time t

        Args:
            x: Input at time t [batch, input_size]
            state: Current hidden state [batch, hidden_size]
            t: Current time (not used directly, but available)

        Returns:
            dhdt: Time derivative of hidden state
        """
        # Get adaptive time constants
        tau = torch.exp(self.tau_log)
        tau = torch.clamp(tau, self.tau_min, self.tau_max)

        # Compute input and recurrent contributions
        input_contrib = self.w_in(x)
        recurrent_contrib = self.w_rec(state)

        # Target activation
        target = torch.tanh(input_contrib + recurrent_contrib + self.bias)

        # LTC dynamics: dh/dt = (-h + f(input + recurrent)) / tau
        dhdt = (-state + target) / tau.unsqueeze(0)

        return dhdt

    def get_time_constants(self) -> torch.Tensor:
        """Return current time constants for analysis"""
        tau = torch.exp(self.tau_log)
        return torch.clamp(tau, self.tau_min, self.tau_max)


class LTCNetwork(nn.Module):
    """
    Full Liquid Time-Constant Network
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        time_constant_range: Tuple[float, float] = (0.1, 10.0),
        ode_method: str = 'dopri5'
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ode_method = ode_method

        # LTC cell
        self.ltc_cell = LTCNeuron(input_size, hidden_size, time_constant_range)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x_sequence: torch.Tensor,
        t_sequence: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence of inputs with continuous-time dynamics

        Args:
            x_sequence: Input sequence [seq_len, batch, input_size]
            t_sequence: Timestamps [seq_len]
            initial_state: Initial hidden state [batch, hidden_size]

        Returns:
            outputs: Output sequence [seq_len, batch, output_size]
            final_state: Final hidden state [batch, hidden_size]
        """
        seq_len, batch_size, _ = x_sequence.shape

        # Initialize state
        if initial_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x_sequence.device)
        else:
            h = initial_state

        outputs = []
        states = []

        for i in range(seq_len):
            # Time span for ODE solver
            if i < seq_len - 1:
                t_span = t_sequence[i:i+2]
            else:
                # For last step, integrate for small duration
                t_span = torch.tensor(
                    [t_sequence[i].item(), t_sequence[i].item() + 0.1],
                    device=t_sequence.device
                )

            # Define ODE function for current input
            current_input = x_sequence[i]

            def ode_func(t, h_t):
                return self.ltc_cell(current_input, h_t, t)

            # Solve ODE to get state evolution
            h_trajectory = odeint(
                ode_func,
                h,
                t_span,
                method=self.ode_method
            )

            # Update state to final time
            h = h_trajectory[-1]

            # Generate output
            output = self.output_layer(h)
            outputs.append(output)
            states.append(h)

        outputs = torch.stack(outputs)

        return outputs, h

    def get_time_constants(self) -> torch.Tensor:
        """Expose time constants for analysis"""
        return self.ltc_cell.get_time_constants()
