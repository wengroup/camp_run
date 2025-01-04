from camp.ase.calculator import CAMPCalculator
from ase.io import read, Trajectory
from ase import units
import sys
from ase.constraints import FixCom


from integrator import NoseHoover
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
import numpy as np


def run_dynamics(
    model_path,
    filepath: str = "config.xyz",
    T: float = 300,
    timestep: float = 1.0,
    ttime: float = 20,
    steps: int = 1000,
    device="cpu",
):
    """
    NVT Nose-Hoover dynamics.

    Args:
        timestep: time step in fs
        T: temp in Kelvin
        ttime:
    """

    calc = CAMPCalculator(model_path, device=device, need_stress=False)

    atoms = read(filepath, format="extxyz")

    # add init velocity
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=T,
        rng=np.random.RandomState(35),  # set random seed for reproducibility
    )

    # remove center of mass momentum
    Stationary(atoms)

    # remove rotational momentum
    ZeroRotation(atoms)

    # add constrains to remove center of mass
    c = FixCom()
    atoms.set_constraint(c)

    atoms.set_calculator(calc)

    # we need to convert units for the integrator.py
    # from line 184: https://github.com/kyonofx/MDsim/blob/ab0a179d03589593ce2b35f4c067fea1ef22444c/simulate.py#L184C4-L186C63
    # done below:
    dyn = NoseHoover(
        atoms,
        timestep * units.fs,
        T * units.kB,
        ttime,
        logfile=f"md-T{T}.log",
    )

    traj = Trajectory(f"md-T{T}.traj", mode="w", atoms=atoms)
    dyn.attach(traj.write, interval=10)

    dyn.run(steps)


if __name__ == "__main__":
    model_path = "/project/wen/commons/projects/camp_mlip/tests/LiPS/model_trainset_19000/checkpoint.ckpt"

    timestep = float(sys.argv[1])
    ttime = float(sys.argv[2])

    run_dynamics(
        model_path,
        filepath="./lips_config.xyz",
        T=520,
        timestep=timestep,
        ttime=ttime,
        steps=int(50000 / timestep),  # to run 50 ps
        device="cuda",
    )
