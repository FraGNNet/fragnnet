import torch
import os
try:
    from lightning.pytorch.profilers.pytorch import PyTorchProfiler, _KINETO_AVAILABLE, _PROFILER, warning_cache, tensorboard_trace_handler, rank_zero_warn, MisconfigurationException, ProfilerAction, ScheduleWrapper, Any
except ModuleNotFoundError:
    from pytorch_lightning.profilers.pytorch import PyTorchProfiler, _KINETO_AVAILABLE, _PROFILER, warning_cache, tensorboard_trace_handler, rank_zero_warn, MisconfigurationException, ProfilerAction, ScheduleWrapper, Any

class MyPyTorchProfiler(PyTorchProfiler):

    def _init_kineto(self, profiler_kwargs: Any) -> None:
        has_schedule = "schedule" in profiler_kwargs
        self._has_on_trace_ready = "on_trace_ready" in profiler_kwargs

        schedule = profiler_kwargs.get("schedule", None)
        if schedule is not None:
            if not callable(schedule):
                raise MisconfigurationException(f"Schedule should be a callable. Found: {schedule}")
            action = schedule(0)
            if not isinstance(action, ProfilerAction):
                raise MisconfigurationException(
                    f"Schedule should return a `torch.profiler.ProfilerAction`. Found: {action}"
                )
        self._default_schedule()
        schedule = schedule if has_schedule else self._default_schedule()
        self._schedule = ScheduleWrapper(schedule) if schedule is not None else schedule
        self._profiler_kwargs["schedule"] = self._schedule

        activities = profiler_kwargs.get("activities", None)
        self._profiler_kwargs["activities"] = activities or self._default_activities()
        self._export_to_flame_graph = profiler_kwargs.get("export_to_flame_graph", False)
        self._export_metrics = profiler_kwargs.get("export_metrics", ("self_cpu_time_total","self_cuda_time_total"))
        # self._metric = profiler_kwargs.get("metric", "self_cuda_time_total")
        with_stack = profiler_kwargs.get("with_stack", False) or self._export_to_flame_graph
        self._profiler_kwargs["with_stack"] = with_stack

    def stop(self, action_name: str) -> None:
        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            del self._recording_map[action_name]

        if not _KINETO_AVAILABLE or self._emit_nvtx:
            return

        if self.profiler is not None and any(action_name.endswith(func) for func in self.STEP_FUNCTIONS):
            assert isinstance(self.profiler, torch.profiler.profile)
            if self._schedule is not None:
                self._schedule.pre_step(action_name)

            # the default schedule requires a minimum of 5 steps to properly work: `wait=1, warmup=1, active=3`.
            # otherwise, this will raise a `segmentation fault`.
            if self._should_override_schedule():
                warning_cache.warn(
                    "The PyTorch Profiler default schedule will be overridden as there is not enough "
                    "steps to properly record traces."
                )
                self._schedule = None
                self.profiler.schedule = torch.profiler.profiler._default_schedule_fn

            def on_trace_ready(profiler: _PROFILER) -> None:
                if self.dirpath is not None:
                    if self._export_to_chrome:
                        handler = tensorboard_trace_handler(
                            str(self.dirpath), self._prepare_filename(action_name=action_name, extension="")
                        )
                        handler(profiler)

                    if self._export_to_flame_graph:
                        for export_metric in self._export_metrics:
                            path = os.path.join(
                                self.dirpath, self._prepare_filename(action_name=action_name+f".{export_metric}", extension=".stack")
                            )
                            print(path, export_metric)
                            profiler.export_stacks(path, metric=export_metric)
                else:
                    rank_zero_warn("The PyTorchProfiler failed to export trace as `dirpath` is None")

            if not self._has_on_trace_ready:
                self.profiler.on_trace_ready = on_trace_ready

            if self._schedule is not None:
                self.profiler.step_num = self._schedule.num_step
            self.profiler.step()
            self.profiler.add_metadata("Framework", "pytorch-lightning")
