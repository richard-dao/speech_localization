import torch
from torch import nn
from snntorch import surrogate

class TernaryLeaky(nn.Module):
    """
    Based on snnTorch Leaky but ternary instead of binary

    First-order leaky integrate-and-fire neuron model with ternary output.
    The neuron outputs -1, 0, or +1 based on the membrane potential.

    Membrane potential decays exponentially with rate beta.
    For :math:`U[T] > U_{\\rm thr+} ⇒ S[T+1] = 1`.
    For :math:`U[T] < U_{\\rm thr-} ⇒ S[T+1] = -1`.

    """

    def __init__(
        self,
        beta,
        threshold_positive=1.0,
        threshold_negative=-1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        super().__init__()

        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=learn_beta)
        self.threshold_positive = nn.Parameter(torch.tensor(threshold_positive), requires_grad=learn_threshold)
        self.threshold_negative = nn.Parameter(torch.tensor(threshold_negative), requires_grad=learn_threshold)

        self.spike_grad = spike_grad if spike_grad is not None else surrogate.atan()
        self.surrogate_disable = surrogate_disable
        self.init_hidden = init_hidden
        self.inhibition = inhibition
        self.reset_mechanism = reset_mechanism
        self.state_quant = state_quant
        self.output = output
        self.graded_spikes_factor = graded_spikes_factor
        self.reset_delay = reset_delay

        self._init_mem()

    def _init_mem(self):
        mem = torch.zeros(0)
        self.register_buffer("mem", mem, False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem

    def forward(self, input_, mem=None):

        if mem is not None:
            self.mem = mem

        if self.init_hidden and mem is not None:
            raise TypeError(
                "`mem` should not be passed as an argument while `init_hidden=True`"
            )

        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)
        
        spk = torch.zeros_like(self.mem)
        abs_prev_spk = torch.abs(spk)
        self.mem = self.beta.clamp(0, 1) * self.mem  * (1 - abs_prev_spk) + input_

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        spk[self.mem > self.threshold_positive] = 1.0
        spk[self.mem < self.threshold_negative] = -1.0

        if not self.reset_delay:
            do_reset = (
                spk / self.graded_spikes_factor - self.reset
            )
            if self.reset_mechanism == "subtract":
                self.mem = self.mem - do_reset * self.threshold_positive
            elif self.reset_mechanism == "zero":
                self.mem = self.mem - do_reset * self.mem

        if self.output:
            return spk, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.mem

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], TernaryLeaky):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Clears hidden state variables to zero."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], TernaryLeaky):
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )
