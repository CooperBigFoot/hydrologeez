import torch

from hydrologeez.models.hbv import HBVForcing, HBVModel


def _model() -> HBVModel:
    return HBVModel(
        tt=torch.tensor(0.0, dtype=torch.float64),
        cfmax=torch.tensor(3.0, dtype=torch.float64),
        sfcf=torch.tensor(1.0, dtype=torch.float64),
        cwh=torch.tensor(0.1, dtype=torch.float64),
        cfr=torch.tensor(0.05, dtype=torch.float64),
        fc=torch.tensor(100.0, dtype=torch.float64),
        lp=torch.tensor(0.7, dtype=torch.float64),
        beta=torch.tensor(2.0, dtype=torch.float64),
        k0=torch.tensor(0.3, dtype=torch.float64),
        k1=torch.tensor(0.1, dtype=torch.float64),
        k2=torch.tensor(0.05, dtype=torch.float64),
        perc=torch.tensor(2.0, dtype=torch.float64),
        uzl=torch.tensor(1.0, dtype=torch.float64),
        maxbas=torch.tensor(3.0, dtype=torch.float64),
    )


def _main_forcing() -> HBVForcing:
    return HBVForcing(
        precip=torch.tensor([[4.0, 6.0, 0.0], [1.0, 8.0, 3.0]], dtype=torch.float64),
        pet=torch.tensor([[0.5, 0.7, 0.8], [0.4, 0.6, 0.7]], dtype=torch.float64),
        temp=torch.tensor([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], dtype=torch.float64),
    )


def _warmup_forcing() -> HBVForcing:
    return HBVForcing(
        precip=torch.tensor([[8.0, 2.0], [2.0, 5.0]], dtype=torch.float64),
        pet=torch.tensor([[0.5, 0.5], [0.4, 0.4]], dtype=torch.float64),
        temp=torch.tensor([[-2.0, 3.0], [1.0, 4.0]], dtype=torch.float64),
    )


def _scalar_parameters(model: HBVModel) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def _per_basin_parameters(parameters: dict[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    return {name: value.expand(batch_size).clone() for name, value in parameters.items()}


def _constant_time_parameters(
    parameters: dict[str, torch.Tensor], batch_size: int, time_steps: int
) -> dict[str, torch.Tensor]:
    return {name: value.expand(batch_size, time_steps).clone() for name, value in parameters.items()}


def test_hbv_run_accepts_time_varying_parameters_during_warmup() -> None:
    model = _model()
    scalar_parameters = _scalar_parameters(model)
    main_parameters = {
        name: value.requires_grad_()
        for name, value in _constant_time_parameters(scalar_parameters, batch_size=2, time_steps=3).items()
    }
    warmup_parameters = {
        name: value.requires_grad_()
        for name, value in _constant_time_parameters(scalar_parameters, batch_size=2, time_steps=2).items()
    }
    main_parameters["fc"] = torch.tensor(
        [[90.0, 100.0, 110.0], [105.0, 115.0, 125.0]], dtype=torch.float64, requires_grad=True
    )
    warmup_parameters["fc"] = torch.tensor([[80.0, 85.0], [110.0, 120.0]], dtype=torch.float64, requires_grad=True)

    observations = model.run(
        _main_forcing(),
        parameters=main_parameters,
        warmup=_warmup_forcing(),
        warmup_parameters=warmup_parameters,
    )

    assert observations.shape == (2, 3)
    observations.sum().backward()
    assert all(value.grad is None for value in warmup_parameters.values())
    fc_gradient = main_parameters["fc"].grad
    assert fc_gradient is not None
    assert torch.isfinite(fc_gradient).all()
    assert torch.count_nonzero(fc_gradient) > 0


def test_constant_time_parameters_match_scalar_and_per_basin_parameters() -> None:
    model = _model()
    scalar_parameters = _scalar_parameters(model)
    per_basin_parameters = _per_basin_parameters(scalar_parameters, batch_size=2)
    main_time_parameters = _constant_time_parameters(scalar_parameters, batch_size=2, time_steps=3)
    warmup_time_parameters = _constant_time_parameters(scalar_parameters, batch_size=2, time_steps=2)
    main = _main_forcing()
    warmup = _warmup_forcing()

    scalar_main = model.run(main, parameters=scalar_parameters)
    per_basin_main = model.run(main, parameters=per_basin_parameters)
    time_varying_main = model.run(main, parameters=main_time_parameters)
    torch.testing.assert_close(per_basin_main, scalar_main, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(time_varying_main, scalar_main, rtol=1e-12, atol=1e-12)

    scalar_warmed = model.run(
        main,
        parameters=scalar_parameters,
        warmup=warmup,
        warmup_parameters=scalar_parameters,
    )
    per_basin_warmed = model.run(
        main,
        parameters=per_basin_parameters,
        warmup=warmup,
        warmup_parameters=per_basin_parameters,
    )
    time_varying_warmed = model.run(
        main,
        parameters=main_time_parameters,
        warmup=warmup,
        warmup_parameters=warmup_time_parameters,
    )
    torch.testing.assert_close(per_basin_warmed, scalar_warmed, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(time_varying_warmed, scalar_warmed, rtol=1e-12, atol=1e-12)
