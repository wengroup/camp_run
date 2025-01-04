"""Nose-Hoover NVT"""

from camp.ase.calculator import CAMPCalculator
from ase.io import read, Trajectory
from ase import units
from ase.md.npt import NPT

from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
import numpy as np


def run_dynamics(
    atoms,
    model_path,
    T: float = 300,
    timestep: float = 0.5,
    steps: int = 1000,
    device="cpu",
):

    calc = CAMPCalculator(model_path, device=device)
    atoms.set_calculator(calc)

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

    delta_t = timestep * units.fs
    tau = 100 * delta_t

    dyn = NPT(
        water,
        timestep=1 * units.fs,
        temperature_K=T,
        ttime=tau,
        pfactor=None,  # pfactor=None makes it Nose--Hoover NVT
        externalstress=0.0,
        logfile=f"md-T{T}.log",
    )

    traj = Trajectory(f"md-T{T}.traj", mode="w", atoms=atoms)
    dyn.attach(traj.write, interval=10)

    dyn.run(steps)


if __name__ == "__main__":

    water = read("liquid-64.xyz", "0")
    water = water.repeat((2, 2, 2))

    model_path = "/project/wen/mjwen/playground/natip_playground/water/20240923-production/job_dir/job_20/240923_water-production/4r9nkh81/checkpoints/epoch=2391-step=856336-bak.ckpt"

    timestep = 1.0
    total_time = 300000
    steps = int(total_time / timestep)
    run_dynamics(
        water,
        model_path,
        T=300,
        timestep=timestep,
        steps=steps,
        device="cuda",
    )
