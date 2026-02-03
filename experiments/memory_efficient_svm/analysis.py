"""
experiments.memory_efficient_svm.analysis

For now, just want to assess the scale of the change in delta for varying dimensions. 
Does this follow correctly from what the paper said it was:

Particle-RW: D
Particle-MALA/GRAD: D^{1/3}
"""

import os

from experiments.memory_efficient_svm.kernels import KernelType
from gradient_csmc.utils.printing import ctext

# run experiments 
KERNELS = (
    KernelType.RW_CSMC,
    # KernelType.TP_CSMC, KernelType.TP_CSMC, 
    # KernelType.MALA_CSMC,
)

STYLES = (
    'na',
    # 'filtering', 'twisted',
    # 'filtering',
)


combination = list(zip(KERNELS, STYLES))

for j in range(len(combination)):
    kernel, style = combination[j]
    exec_str = (
        "python3 distribute_grid.py --delta-scaling "
        "--kernel {} --style {} "
    )
    exec_str = exec_str.format(kernel.value, style)
    print("\nExcecuting: ", ctext(exec_str, "red"))
    os.system(exec_str)