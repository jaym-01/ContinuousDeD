""" Simple subclassing of NeuralCDE overwriting the vector field. """

import abc

import torch
import torchcde
from torch import nn

from base_vector_field import OriginalVectorField as Vector_Field
from ncde_utils import create_net

SPLINES = {
    "linear": torchcde.LinearInterpolation,
    "rectilinear": torchcde.LinearInterpolation
}

class NeuralCDE(nn.Module, abc.ABC):

    """Meta class for Neural CDE modelling.

    Attributes:
        nfe (int): Number of function evaluations, this is inherited from the vector field if implemented
    """

    def __init__(
                self,
                input_dim,
                hidden_dim,
                output_dim,
                static_dim=None,
                action_dim=None,
                hidden_hidden_dim=16,
                num_layers=3,
                pred_num_layers=2,
                pred_num_units=100,
                vector_field_type="matmul",
                use_initial=True,
                interpolation="rectilinear",
                adjoint=True,
                solver="rk4",
                return_sequences=False,
                apply_predictor=True,
                return_filtered_rectilinear=True,
                device='cpu'
    ):
        """
        Args:
            input_dim (int): The dimension of the path. (Number of covariates+intensity+action)
            hidden_dim (int): The dimension of the Vector Field hidden state.
            output_dim (int): The dimension of the output of the predictor. (Number of continuously varying covariates)
            static_dim (int): The dimension of any static values, these will be concatenated to the initial values and
                put through a network to build h0.
            action_dim (int): The dimension of the actions, these will be concatenated to the initial values and put through a network to build h0.
            hidden_hidden_dim (int): The dimension of the hidden layer in the RNN-like block.
            num_layers (int): The number of hidden layers in the vector field. Set to 0 for a linear vector field.
                net with the given density. Hidden and hidden hidden dims must be multiples of 32.
            pred_num_layers (int): The number of hidden layers in the MLP prediction network used to construct the prediction
                of the next observation from the hidden state
            pred_num_units (int): The dimension of the hidden layers of the prediction MLP
            vector_field_type (str): One of ('matmul', 'evaluate', 'derivative'; default='matmul') determines whether the vector field
                will apply [f(h) dX/dt, f(h, X), f(h, dX/dt)]
            use_initial (bool): Set True to use the initial absolute values to generate h0.
            interpolation (str): Interpolation method from ('linear', 'rectilinear'; default='rectilinear').
            adjoint (bool): Set True to use odeint_adjoint.
            solver (str): ODE solver, must be implemented in torchdiffeq.
            return_sequences (bool): If True will return the linear function on the final layer, else linear function on
                all layers.
            apply_predictor (bool): Set False for no final prediction model to be applied to the hidden state.
            return_filtered_rectilinear (bool): Set True to return every other output if the interpolation scheme chosen
                is rectilinear, this is because rectilinear doubles the input length. False will return the full output.
            device (str): Whether we'll be processing the model (and data) on 'cpu' or GPU ('cuda')
        """
        super().__init__()

        self.input_dim = input_dim
        self.static_dim = static_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.num_layers = num_layers
        self.pred_num_layers = pred_num_layers
        self.pred_num_units = pred_num_units
        self.use_initial = use_initial
        self.interpolation = interpolation
        self.adjoint = adjoint
        self.solver = solver
        self.return_sequences = return_sequences
        self.apply_predictor = apply_predictor
        self.return_filtered_rectilinear = return_filtered_rectilinear
        self.vector_field_type = vector_field_type
        self.device=device


        # Set initial linear layer
        if self.initial_dim > 0:
            self.initial_linear = nn.Linear(self.initial_dim, self.hidden_dim)

        # Spline
        assert (
            self.interpolation in ["linear", "rectilinear"]
        ), "Unrecognised interpolation scheme {}".format(self.interpolation)
        self.spline = SPLINES.get(self.interpolation)

        # Set options
        assert self.solver in ["rk4", "dopri5"]
        self.atol = 1e-5
        self.rtol = 1e-3
        self.cdeint_options = (
            {"step_size": 1} if self.solver == "rk4" else {"min_step": 0.5}
        )

        # The net that is applied to h_{t-1}
        self.func = Vector_Field(
            input_dim=self.input_dim+self.action_dim,  # Account for using previous action as a dimension of the input
            hidden_dim=self.hidden_dim,
            hidden_hidden_dim=self.hidden_hidden_dim,
            num_layers=self.num_layers,
            sparsity=None,
            vector_field_type=vector_field_type,
        )

        # Linear classifier to apply to final layer
        # TODO... Fix up general function class... for prediciton module...
        self.predictor = (
            create_net(self.hidden_dim, self.output_dim, self.pred_num_layers, self.pred_num_units)
            if apply_predictor
            else lambda x: x
        )

    @property
    def initial_dim(self):
        # Setup initial dim dependent on `use_initial` and `static_dim` options
        initial_dim = 0
        if self.use_initial:
            initial_dim += self.input_dim
        if self.static_dim is not None:
            initial_dim += self.static_dim
        initial_dim += self.action_dim
        return initial_dim

    @property
    def nfe(self):
        nfe_ = None
        if hasattr(self.func, "nfe"):
            nfe_ = self.func.nfe
        return nfe_

    def _setup_h0(self, inputs):
        """Sets up the initial value of the hidden state.
        The hidden state depends on the options `use_initial` and `static_dim`. If either of these are specified the
        hidden state will be generated via a network applied to either a concatenation of the initial and static data,
        or a network applied to just initial/static depending on options. If neither are specified then a zero initial
        hidden state is used.
        """
        # Split out the components of the inputs
        static, temporal, actions = inputs
        
        # We'll append the previous action to the observation to help form an information state
        ac_shifted = torch.cat((torch.zeros(actions.shape[0], 1, actions.shape[-1]).to(self.device), actions[:, :-1, :]), dim=1)

        # We double up the temporal dimension to match the shape of the temporal data (that was rectilinearly interpolated)
        if self.interpolation == 'rectilinear':
            ac_shifted = ac_shifted.repeat_interleave(2, 1)[:,:-1, :]
        
        # Compute the spline from the temporal+prev. action data
        spline = self.spline(torch.cat((temporal, ac_shifted), dim=-1))

        if static is not None:
            if self.use_initial:
                h0 = self.initial_linear(
                    # Concatenate the static data to the 
                    torch.cat((static, spline.evaluate(0)), dim=-1)
                )
            else:
                h0 = self.initial_linear(static)
        else:
            if self.use_initial:
                h0 = self.initial_linear(spline.evaluate(0))
            else:
                h0 = torch.autograd.Variable(
                    torch.zeros(inputs.size(0), self.hidden_dim)
                ).to(inputs.device)
        

        return spline, h0

    def _make_outputs(self, hidden):
        """Hidden state to output format depending on `return_sequences` and rectilinear (return every other)."""
        if self.return_sequences:
            outputs = self.predictor(hidden)

            # If rectilinear and return sequences, return every other value
            if (
                self.interpolation == "rectilinear"
            ) and self.return_filtered_rectilinear:
                outputs = outputs[:, ::2]
                hidden = hidden[:, ::2]
        else:
            outputs = self.predictor(hidden[:, -1, :])
            hidden = hidden[:, -1, :]
        return outputs, hidden

    def calculate_loss(self, pred, target, mask):
        """ Compute the loss (MSE loss for now). 
        
        To (hopefully) stabilize learning, we'll only compute loss for the dimensions 
        that were actually observed at each time point. We use the mask (1/0 if covariate was/wasn't observed)
        to isolate those features at each time. 
        """

        return nn.functional.mse_loss(target * mask, pred * mask)
        

    def forward(self, inputs):

        # Handle h0 and inputs
        spline, h0 = self._setup_h0(inputs)

        # Only return sequences with a fixed grid solver
        if self.return_sequences:
            # assert (
            #     self.solver == "rk4"
            # ), "return_sequences is only allowed with a fixed grid solver (for now)"
            times = spline.grid_points
        else:
            times = spline.interval

        hidden = torchcde.cdeint(
            spline,
            self.func,
            h0,
            t=times,
            adjoint=self.adjoint,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
            options=self.cdeint_options,
        )

        # Convert to outputs
        return self._make_outputs(hidden)