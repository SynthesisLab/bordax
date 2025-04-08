We use the following conventions. 

In `policies\[architecture].py`, where `[architecture]` is a type of policy (such as `dtsemnet` or `mlp`), the methods for creating and using a policy during training. 
For example, `make_policy_value_mlp` returns a `PolicyValue`-tuple, where `Policy` has two methods: `init` and `get_distribution`.
Method `init` takes a random key and returns initial parameters of the model.
Method `get_distribution` takes parameters of the policy and an observation, and returns a probabilistic distribution together with additional information.
Notice that it does not return an action.

`policies\utils.py` contains a function `action_value_factory` that takes a `PolicyValue`-tuple and returns a pair of functions: `make_policy` and `make_value`.
These functions are used one does not need access to raw outputs of the model, for example, while collecting a rollout or evaluating a model.
`make_policy` takes policy parameters and returns the action function.