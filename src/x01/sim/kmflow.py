import math

import torch


class KMFlowSolver:
    def __init__(
        self,
        N: int,
        re: float,
        alpha: float,
        tau: float,
        dt: float,
        device: torch.device,
        forcing_n: int = 4,
        drag: float = 0.1,
    ):  # pseudo-spectral 2D spectral Kolmogorov flow solver
        self.N = N
        self.nu = 1.0 / re
        self.dt = dt
        self.device = device
        self.alpha = alpha
        self.tau = tau
        self.forcing_n = forcing_n
        self.drag = drag

        kx = torch.fft.fftfreq(N, d=1.0 / N).to(device)
        ky = torch.fft.rfftfreq(N, d=1.0 / N).to(device)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        K2 = KX**2 + KY**2

        self.KX = KX
        self.KY = KY
        self.K2 = K2

        inv_lap = torch.zeros_like(K2)
        inv_lap[K2 > 0] = 1.0 / K2[K2 > 0]
        self.inv_lap = inv_lap

        self.dealias = (K2 <= (N / 3.0) ** 2).to(torch.float32)
        self.dealias[0, 0] = 0.0

        self.forcing_hat = torch.zeros(N, N // 2 + 1, dtype=torch.complex64, device=device)
        if forcing_n > 0:
            self.forcing_hat[0, forcing_n] = -float(forcing_n) * (N**2) / 2.0

        sigma = self.tau ** (alpha - 1.0)
        sqrt_eig = (N**2) * math.sqrt(2.0) * sigma * (K2 + tau**2) ** (-alpha / 2.0)
        sqrt_eig[0, 0] = 0.0
        self.sqrt_eig = sqrt_eig

        self.omega_hat: torch.Tensor | None = None

    def reset(self, seeds: list[int]) -> None:
        batch = []
        for seed in seeds:
            gen = torch.Generator(device=self.device).manual_seed(seed)
            xi_re = torch.randn(self.N, self.N // 2 + 1, device=self.device, generator=gen)
            xi_im = torch.randn(self.N, self.N // 2 + 1, device=self.device, generator=gen)
            omega_hat = (xi_re + 1j * xi_im) * self.sqrt_eig
            omega_hat = omega_hat * self.dealias
            batch.append(omega_hat)

        self.omega_hat = torch.stack(batch)

    def step(self) -> None:
        cn = 0.5 * self.dt * (self.nu * self.K2 + self.drag)
        cn_m = 1.0 - cn
        cn_p_inv = 1.0 / (1.0 + cn)

        rhs1 = self._rhs(self.omega_hat)
        omega_tilde = (cn_m * self.omega_hat + self.dt * rhs1) * cn_p_inv

        rhs2 = self._rhs(omega_tilde)
        self.omega_hat = ((cn_m * self.omega_hat + self.dt * 0.5 * (rhs1 + rhs2)) * cn_p_inv) * self.dealias

    def _rhs(self, omega_hat: torch.Tensor) -> torch.Tensor:
        psi_hat = omega_hat * self.inv_lap
        u_hat = 1j * self.KY * psi_hat
        v_hat = -1j * self.KX * psi_hat
        dw_dx_hat = 1j * self.KX * omega_hat
        dw_dy_hat = 1j * self.KY * omega_hat

        u = torch.fft.irfft2(u_hat, s=(self.N, self.N))
        v = torch.fft.irfft2(v_hat, s=(self.N, self.N))
        dw_dx = torch.fft.irfft2(dw_dx_hat, s=(self.N, self.N))
        dw_dy = torch.fft.irfft2(dw_dy_hat, s=(self.N, self.N))

        advection_hat = torch.fft.rfft2(u * dw_dx + v * dw_dy)
        return -advection_hat + self.forcing_hat

    def get_vorticity(self) -> torch.Tensor:
        return torch.fft.irfft2(self.omega_hat, s=(self.N, self.N))
