"""DynaCU-Bench's own agent protocol (AOI `standard` mode), with only the model call made cancellable.

Observation, prompt and parsing come from the pinned AOI checkout unchanged:
* the context text is `ObservationRecord.to_prompt_text()` over AOI's `TrajectoryStore(context_depth=5)`;
* the images are `ObservationRecord.get_images()` (standard mode: the current screenshot, no keyframes);
* the request is exactly `LocalVLLMModel(generic)`'s: [context text, images..., "Task: <instr>" + NARRATION_INSTRUCTION];
* the reply is parsed by AOI's `_parse_output` (NARRATION / ACTION);
* a failed action's error is fed back on the next step as AOI does ([ACTION ERROR ...]).
`step_counter` is the AOI step number; interrupted / invalidated generations do not advance it.
"""

from __future__ import annotations

import io
import time

from nightwatch.agents.base import AgentAdapter, Prediction
from nightwatch.agents.llm import CallStats, Cancelled, Endpoint, stream_chat


class AOIAgentAdapter(AgentAdapter):
    def __init__(self, agent_cfg):
        from nightwatch.envs.dynacu import _aoi_imports

        _aoi_imports()
        from aoi.cu_model import NARRATION_INSTRUCTION, _parse_output
        from aoi.observation_record import ObservationRecord, TrajectoryStore

        self._OR, self._TS, self._parse, self._instr = ObservationRecord, TrajectoryStore, _parse_output, NARRATION_INSTRUCTION
        ep = Endpoint.from_env("AGENT")
        for k in ("model", "max_tokens", "temperature", "top_p"):
            if agent_cfg.get(k) is not None:
                setattr(ep, k, agent_cfg.get(k))
        if agent_cfg.get("max_tokens") is None:
            ep.max_tokens = 1024  # LocalVLLMModel default
        self.ep = ep
        self.supports_cancel = ep.stream and bool(agent_cfg.get("cancellable", True))
        self.name = f"aoi_standard:{ep.model}"
        self.reset()

    def reset(self, vm_ip=None) -> None:
        self.traj = self._TS(context_depth=5, screenshot_history=5)
        self.step = 0
        self.last_error = None

    def on_action_result(self, info: dict) -> None:
        """A failed action (AOI's own feedback) or one the harness withheld because the screen changed: the next
        prompt carries it as [ACTION ERROR], exactly the channel AOI uses for failed actions."""
        if info and info.get("ok") is False and info.get("error"):
            self.last_error = info["error"]

    def predict(self, instruction, obs, cancel) -> Prediction:
        from PIL import Image

        t = time.perf_counter()
        step = self.step + 1
        shot = Image.open(io.BytesIO(obs["screenshot"])).convert("RGB")
        rec = self._OR(step_id=step, context_steps=self.traj.get_context(step), current_audio_text="",
                       current_audio_context="", keyframes=[], post_action_screenshot=shot, task_instruction=instruction)
        text = rec.to_prompt_text()
        if self.last_error:
            text += (f"\n\n[ACTION ERROR — previous action failed]\n  Error: {self.last_error}\n"
                     "  Try a different action. Use click(x, y) with pixel coordinates. To type text, use type(\"text\") or "
                     "click the input first then type(\"text\"). See [PAGE ELEMENTS] above for valid element IDs.")
        from nightwatch.utils import b64

        content = [{"type": "text", "text": text}] if text.strip() else []
        for img in rec.get_images():
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(buf.getvalue())}})
        content.append({"type": "text", "text": f"Task: {instruction}{self._instr}"})
        st = CallStats()
        try:
            raw = stream_chat(self.ep, [{"role": "user", "content": content}], cancel if self.supports_cancel else None, st)
        except Cancelled:
            return Prediction(cancelled=True, seconds=time.perf_counter() - t, output_tokens=st.output_tokens,
                              input_tokens=st.input_tokens, tokens_estimated=True, t_first_token=st.t_first_token,
                              extra={"http": st.http})
        except Exception as e:
            raw = ""
            err = f"{type(e).__name__}: {e}"
        else:
            err = ""
        out = self._parse(raw) if raw else None
        action = out.action if out else "wait"  # AOI: a failed model call becomes wait
        self.last_error = None
        self.traj.append(step_id=step, step_start_time=time.time(), step_end_time=time.time(), audio_text="",
                         visual_narration=out.narration if out else "Model error", action=action, n_keyframes=0,
                         audio_model_called=False, screenshot=shot)
        self.step = step
        return Prediction(response=raw, actions=[action], seconds=time.perf_counter() - t, error=err,
                          t_first_token=st.t_first_token, input_tokens=st.input_tokens, output_tokens=st.output_tokens,
                          tokens_estimated=st.estimated, retries=st.retries, extra={"http": st.http})

    def snapshot(self):
        return (list(self.traj._steps), list(self.traj._screenshots), self.step, self.last_error)

    def restore(self, snap) -> None:
        self.traj._steps, self.traj._screenshots, self.step, self.last_error = list(snap[0]), list(snap[1]), snap[2], snap[3]
