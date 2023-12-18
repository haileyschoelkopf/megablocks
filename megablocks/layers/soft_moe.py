from megablocks.layers import common
from megablocks.layers import mpu
from megablocks.layers import mlp
from megablocks.layers import moe
from megablocks.layers.all_to_all import all_to_all
from megablocks.layers.arguments import Arguments
import megablocks.ops as ops
import numpy as np
import torch


# TODO: add a similar identity fn router for soft MoE

# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.

# (hailey): in Soft MoE we assume a uniform unchanging expert assignment of slots to experts. 
class _UniformExpertAssignment(torch.autograd.Function):


    @staticmethod
    def forward(ctx, x, num_experts):
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)
_uniform_expert_assignment = _UniformExpertAssignment.apply


class LearnedSoftRouter(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args

        # Learned soft MoE router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert model
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            args.hidden_size,
            args.moe_num_experts * args.moe_num_slots_per_expert, # TODO: add this arg to the arguments.py file
            bias=False,
            dtype=common.dtype(args), # TODO: why are routers not in fp32?
            device=args.device)
        args.init_method(self.layer.weight)

    def jitter(self, x): # TODO: re-check jitter (probably no change needed) adapt jitter to soft routing
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def forward(self, x): # TODO: change return signature of forward() to return combine + dispatch weights
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        sl, bs, hs = x.size()
        # get router scores 
        # x: [sl, bs, hs] -> view as [sl * bs, hs]
        # scores: [sl * bs, n_experts * n_slots_per_expert]
        # (as opposed to [sl * bs, n_experts] in typical SparseMoE case)
        scores = self.layer(x.view(-1, x.shape[-1])).view(sl, bs, -1) # TODO: get rid of this view() ?

        # combine weights: softmax over output *slots*
        combine_weights = scores.softmax(dim=2) # combine_weights: [sl, bs, n_experts * n_slots_per_expert]

        # dispatch weights: softmax over input *tokens* within a sequence
        dispatch_weights = scores.softmax(dim=0) # dispatch_weights: [sl, bs, n_experts * n_slots_per_expert]
                       
        expert_indices = ( # TODO: is it ok to manually create these expert indices? do we need the _uniform_expert_assignment? is this the right dtype to create in?
            _uniform_expert_assignment(torch.remainder(torch.arange((sl * bs), dtype=torch.int, device=dispatch_weights.device), self.args.moe_num_experts).view((sl, bs)), self.args.moe_num_experts) 
            # TODO: add a check earlier on that returns identity for combine/dispatch weights optionally, for testing purposes
        )

        # workaround: create dummy all-ones expert weights, since each slot only goes to one expert. 
        # TODO: omit expert_weights entirely if possible 
        expert_weights = torch.ones((sl * bs, 1), dtype=dispatch_weights.dtype, device=dispatch_weights.device)

        return scores, expert_weights, combine_weights, dispatch_weights, expert_indices


# TODO: remove class methods unchanged from ParallelMLP

class ParallelSoftMLP(moe.ParallelMLP):

    def __init__(self, args : Arguments):
        super(ParallelSoftMLP, self).__init__(args)
       
        assert args.moe_capacity_factor == 1, "non-1.0 capacity factors disallowed for Soft MoE"
        assert args.moe_top_k == 1

        self.num_slots_per_expert = self.args.moe_num_slots_per_expert

    def expert_capacity(self, tokens): # TODO: factor in num_slots_per_expert explicitly?
        world_size = mpu.get_expert_parallel_world_size(self.args)
        tokens_per_expert = (
            tokens * world_size / self.num_experts)
        return int(tokens_per_expert)

    # def load_balancing_loss(self, tokens_per_expert, expert_scores): # TODO: any good analogue to load balancing loss in Soft MoE?
    #     """Calculate the load balancing loss contribution."""
    #     assert len(expert_scores.size()) == 2
    #     tokens, num_experts = expert_scores.size()
    #     assert num_experts == self.num_experts
    #     assert len(tokens_per_expert.size()) == 1
    #     num_experts, = tokens_per_expert.size()
    #     assert num_experts == self.num_experts
    #     scale = self.num_experts / (tokens * self.top_k)
    #     return scale * torch.dot(
    #         tokens_per_expert.to(expert_scores.dtype),
    #         expert_scores.mean(dim=0))

    def indices_and_bins(self, top_expert):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    def permute_and_compute(
            self,
            x,
            tokens_per_expert, # unused
            indices,
            bin_ids, # unused
            expert_weights,
            bins,
            expert_capacity,
            top_k):
        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.binned_gather(
            x, indices, bins, expert_capacity, top_k)

        # Perform the expert computation. Note that we don't
        # use biases for these linear operations.
        x = self.mlp(x)

        # Un-route the data for the MoE output.
        return ops.binned_scatter(
            x, indices, expert_weights, bins, top_k)

    def forward_once(self, x, expert_weights, top_experts):
      
        # x: [sl, bs, hs]
        # expert_weights: [sl * bs, top-k]
        # top_experts: [sl * bs, top-k]
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_experts))

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            sl, bs, hs = x.size()
            expert_capacity = self.expert_capacity(sl * bs)
            if expert_capacity == 0:
                expert_capacity = torch.max(tokens_per_expert).item()

        x = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            expert_capacity,
            self.top_k)
        return x, tokens_per_expert

    def parallel_forward_once(self, x, expert_weights, top_experts):
        # TODO: skip permuting + unpermuting?
      
        # NOTE: This function implements the same computation as forward_once
        # but with expert model parallelism.
        #
        # 1. Permute the tokens locally so that they are grouped by their
        # expert assignments. This allows us to transfer all of the tokens
        # for a remote device in one communication primitive.
        #
        # 2. Permute the tokens across the expert parallel devices. After
        # this is completed each device has all of the tokens assigned to
        # its set of experts in its local HBM.
        #
        # 3. Permute the tokens locally so that they are grouped by their
        # expert assignement. After the distributed permutation the tokens
        # are grouped by which device they came from. We re-order them
        # locally to allow for efficient computation.
        #
        # After this series of permutations we compute the linear layers
        # and then repeat these three steps in reverse to produce the final
        # output.
        #
        # Compute the mapping of local tokens to experts.
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_experts))

            # If we're sharding the experts along the hidden dimension
            # multiple devices own parts of the same sets of experts.
            # Replicate the token counts so every device gets the counts.
            repeated_tokens_per_expert = ops.repeat(
                tokens_per_expert, (mpu.hidden_sharding_degree(self.args),))

            # Pass token count information to the device on which the
            # target expert resides.
            parallel_tokens_per_expert = torch.empty_like(repeated_tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert,
                repeated_tokens_per_expert,
                group=self.args.expert_parallel_group,
                async_op=True)

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        #
        # This view updates the shape of the tensor from [sl, bs, hs] to
        # [sl * bs, hs] prior to the permutation.
        
        x = x.reshape(-1, x.shape[-1]) # TODO: why does this need to be a reshape. It shouldn't have to be (I hope)
        x = ops.gather(
            x,
            indices,
            bin_ids,
            bins,
            self.top_k)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()
            experts_per_rank = mpu.experts_per_rank(self.args)

            # Reshape to [world_size, num_experts_per_rank].
            world_size = mpu.get_expert_parallel_world_size(self.args)
            repeated_tokens_per_expert = (
                repeated_tokens_per_expert.view(world_size, experts_per_rank))
            parallel_tokens_per_expert = (
                parallel_tokens_per_expert.view(world_size, experts_per_rank))

            # TODO(tgale): It might be faster to do this on the GPU and
            # then communicate the results back to the host.
            send_counts = repeated_tokens_per_expert.cpu().sum(dim=-1)
            parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
            recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

        # If we're sharding the experts along the hidden dimension
        # multiple devices own parts of the same sets of experts.
        # Replicate the token counts so devices that share experts
        # get all of the tokens assigned to them.
        #
        # TODO(tgale): Fuse this into the prior, local permutation.
        x = ops.repeat(x, (mpu.hidden_sharding_degree(self.args), 1))

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        parallel_x, parallel_x_handle = all_to_all(
            x, recv_counts, send_counts,
            self.args.expert_parallel_group,
            async_op=True)

        with torch.no_grad():
            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.
            replicate_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert.flatten(), 0)
            replicate_bins = (
                replicate_bins.view(1)
                if not len(replicate_bins.size())
                else replicate_bins
            )

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(
                    self.num_experts * mpu.hidden_sharding_degree(self.args),
                    dtype=torch.int32,
                    device=indices.device
                ),
                mpu.experts_per_rank(self.args),
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0),
                replicate_bins, tokens_received).flatten()

            # TODO(tgale): The sort_end_bit here can be reduced.
            parallel_bin_ids, parallel_indices = ops.sort(
                parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(
                dim=0, dtype=torch.int)
            parallel_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert, 0)
            parallel_bins = (
                parallel_bins.view(1)
                if not len(parallel_bins.size())
                else parallel_bins
            )

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, hs = x.size()
            expert_capacity = self.expert_capacity(tokens)
            if expert_capacity == 0:
                expert_capacity = torch.max(
                    parallel_tokens_per_expert).item()

        # Locally permute the tokens and perform the expert computation.
        # Block to make sure that the cross-device permutation is complete.
        if isinstance(self.mlp, mlp.GroupedMLP):
            # GroupedMLP requires counts on CPU. We can use the tensor already
            # moved to CPU for the prior all_to_all, which avoids an extra
            # device synchronization.
            parallel_tokens_per_expert = parallel_tokens_per_expert_cpu.sum(
                dim=0, dtype=torch.int)
        parallel_x_handle.wait()
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            None,  # expert_weights
            parallel_bins,
            expert_capacity,
            top_k=1)

        # Un-permute the tokens across the devices.
        x, _ = all_to_all(
            parallel_x, send_counts, recv_counts,
            self.args.expert_parallel_group)

        # Reduce along the hidden sharding to get the final outputs.
        #
        # TODO(tgale): Fuse this into the following local permutation.
        shape = (
            mpu.hidden_sharding_degree(self.args),
            -1,
            self.args.hidden_size
        )
        x = ops.sum(x.view(shape), dim=0)

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.top_k,
            self.args.quantize_scatter_num_bits)
        return x, tokens_per_expert.flatten()

    def forward(self, x, scores, expert_weights, top_experts):
        in_shape = x.size()

        # Compute the experts.
        x, tokens_per_expert = self.forward_fn(
            x, expert_weights, top_experts)
        # save_load_balancing_loss((tokens_per_expert, scores))
        x = x.view(in_shape)
        if self.bias is not None:
            if self.args.return_bias:
                return x, self.bias
            return x + self.bias
        return x


class SoftMoE(moe.MoE):

    def __init__(self, args : Arguments):
        # we don't want to call moe.MoE's super() method.
        super(moe.MoE, self).__init__()

        self.args = args

        # Token router.
        self.router = LearnedSoftRouter(args) # TODO: check router logic is correct.

        # Expert computation helper.
        self.experts = ParallelSoftMLP(args)

    def dispatch(self, x, dispatch_weights):
        # takes in input tokens and returns linearly-combined inputs to SoftMoE's slots.

        # x: [sl, bs, hs]
        # TODO: currently assumes same shape returned as input. we will need to handle total_slots < seqlen case
        input_slots = torch.einsum('n b h, n b s -> s b h', x, dispatch_weights) # n = seqlen, b = batchsize, h = hiddensize, s = totalslots
        assert input_slots.shape == x.shape # temporary assert to check output shape. once general total slot count supported should not hold

        # input_slots: [sl, bs, hs]
        return input_slots

    def combine(self, x, combine_weights):
        # takes in outputs from slots and returns a linear combination of slot outputs for each *token*.
        
        # x: [total_slots, bs, hs] # TODO: for now we assume total_slots = sl
        # combine_weights: [sl, bs, n_experts * n_slots_per_expert]
        output_tokens = torch.einsum("s b h, n b s -> n b h", x, combine_weights) # n = seqlen, b = batchsize, h = hiddensize, s = totalslots

        assert output_tokens.shape == x.shape # assert to check output shape
        # output_tokens: [sl, bs, hs]
        return output_tokens

    def forward(self, x):
        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth.
        x = common.cast_if_autocast_enabled(x)

        assert self.args.moe_num_experts * self.args.moe_num_slots_per_expert == x.shape[0], "Currently only support num_slots == seqlen case"

        # TODO: is it safe to do combine/dispatch on each rank locally? then do forward() logic as is typical?

        # TODO: compute the l2 normalization used
        # TODO: compute the balance / router statistics from paper's code release
        
        # Compute the expert scores and assignments.
        scores, expert_weights, combine_weights, dispatch_weights, top_experts = self.router(x) # TODO: return from my implemented router not yet matching this

        # Slots = linear combination of tokens, given by dispatch weights
        x = self.dispatch(x, dispatch_weights)

        # Compute the experts.
        experts, _ = self.experts(x, scores, expert_weights, top_experts)

        # Output tokens = linear combination of slots, given by combine weights
        return self.combine(x, combine_weights), _
        
