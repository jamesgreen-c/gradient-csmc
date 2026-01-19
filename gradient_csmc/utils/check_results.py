import os


def check_results_path(dirpath):
    if not os.path.exists(dirpath):
            raise FileNotFoundError(
        f"""

        No such experiment found with the following configuration

        Kernel: {kernel.name},
        N Samples: {args.n_samples}, 
        M: {args.M}, 
        T: {T}, 
        D: {args.D}, 
        N Factors: {args.n_factors}, 
        N Particles: {args.N}, 
        style: {style}, 
        Target Alpha: {TARGET_ALPHA}, 
        BPF Init: {args.bpf_init},
        Resampling Type: {args.resampling},
        Backward Sampling: {args.backward},
        Seed: {args.seed}
        """
            )