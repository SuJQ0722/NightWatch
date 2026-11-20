"""OSWorld's Qwen3.5-VL agent (branch dev_djlu/qwen35vl_agent), unchanged except for its transport.

We subclass `mm_agents.qwen35vl_agent.Qwen35VLAgent` from the OSWorld checkout and override only `call_llm`: same prompt,
screenshot resize, 0-999 relative coordinates, history folding and parsing, but the request is streamed and can be
cancelled. The defaults below are the branch's own (run_multienv_qwen35vl.py); `agent.*` in the config overrides them.
AGENT_MAX_IMAGES applies an endpoint's image limit (e.g. 4), as for the PromptAgent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time

from nightwatch.agents.base import AgentAdapter, Prediction
from nightwatch.agents.llm import CallStats, Cancelled, Endpoint, stream_chat
from nightwatch.agents.prompt_agent import cap_image_bytes, downscale_obs, fit_image_budget, rescale_coordinates, with_aoi_observation

# AOI's narration request (aoi/cu_model.py NARRATION_INSTRUCTION), with this agent's Action + tool call instead of code
AOI_NARRATION_TOOL = ("Also provide a one-sentence visual narration of what is NEW on screen, in the keyframes and the "
                      "current screenshot. Start your reply with a line\nNARRATION: <one sentence about new visual info, or "
                      "\"No visual change.\">\nthen give your Action and tool call as usual.")


def _make_class():
    from mm_agents.qwen35vl_agent import Qwen35VLAgent  # from third_party/OSWorld (on sys.path via osworld.pth)

    class StreamingQwen35Agent(Qwen35VLAgent):
        endpoint: Endpoint
        cancel: threading.Event | None = None
        stats: CallStats | None = None
        cancelled = False
        native_tools = False  # send `tools` and get the call back as tool_calls (needed on some servers, see tool_calls_as_xml)
        max_images = None
        max_image_chars = 24_000_000  # base64 chars of all images in one request (OpenRouter 413 above 30MB)
        aoi_keyframes: list | None = None  # set per call by the aoi_visual baseline (None = this agent unchanged)

        def call_llm(self, payload, model):  # noqa: ARG002 - same contract as Qwen35VLAgent.call_llm
            max_images = self.max_images
            if self.aoi_keyframes is not None:  # the aoi_visual baseline: keyframes before the current screenshot
                messages = with_aoi_observation(payload["messages"], self.aoi_keyframes, max_images,
                                                narration=AOI_NARRATION_TOOL)
            else:
                messages, _ = fit_image_budget(payload["messages"], [], max_images)
            messages, dropped = cap_image_bytes(messages, self.max_image_chars)
            if self.stats is not None:
                self.stats.images_capped = dropped
            kw = {}
            if self.native_tools:  # the same tool definition the system prompt lists
                kw["tools"] = [json.loads(re.search(r"<tools>\n(.*?)\n</tools>", messages[0]["content"][0]["text"], re.S)[1])]
            try:
                return stream_chat(self.endpoint, messages, self.cancel, self.stats, max_tokens=payload["max_tokens"],
                                   temperature=payload["temperature"], top_p=payload["top_p"], **kw)
            except Cancelled:  # the loop discards this prediction and restores the agent's history
                self.cancelled = True
                return "WAIT"

    return StreamingQwen35Agent


class Qwen35AgentAdapter(AgentAdapter):
    def __init__(self, agent_cfg):
        cls = _make_class()
        ep = Endpoint.from_env("AGENT")
        if "extra_body" in agent_cfg:
            ep.extra_body = dict(agent_cfg["extra_body"])
        ep.retries = int(agent_cfg.get("retries", ep.retries))
        self.agent = cls(model=ep.model, max_tokens=int(agent_cfg.get("max_tokens", 32768)),
                         top_p=float(agent_cfg.get("top_p", 0.9)), temperature=float(agent_cfg.get("temperature", 0.0)),
                         history_n=int(agent_cfg.get("history_n", 100)),
                         coordinate_type=agent_cfg.get("coordinate_type", "relative"),
                         image_max=int(agent_cfg.get("image_max", 20)), fold_size=int(agent_cfg.get("fold_size", 10)))
        self.agent.endpoint = ep
        self.agent.native_tools = bool(agent_cfg.get("native_tools", False))
        if "max_image_mb" in agent_cfg:  # None/0 = no size cap
            self.agent.max_image_chars = int(float(agent_cfg["max_image_mb"] or 0) * 1e6) or None
        self.agent.max_images = int(agent_cfg.get("max_images") or os.environ.get("AGENT_MAX_IMAGES") or 0) or None
        # e.g. [1280, 720] with coordinate_type absolute: the model sees this size and its clicks are mapped back to the
        # screen (MiMo answers 1920x1080 screenshots in 1280x720 pixels, also when asked for 0-999 coordinates)
        self.obs_size = tuple(agent_cfg.get("obs_size") or ()) or None
        self.supports_cancel = ep.stream and bool(agent_cfg.get("cancellable", True))
        self.name = f"qwen35_agent:{ep.model}"

    def reset(self, vm_ip=None) -> None:
        self.agent.reset(logging.getLogger("nightwatch.agent"), vm_ip=vm_ip)

    def predict(self, instruction, obs, cancel) -> Prediction:
        a = self.agent
        a.cancel = cancel if self.supports_cancel else None
        a.stats = CallStats()
        a.cancelled = False
        a.aoi_keyframes = obs.get("aoi_keyframes")
        obs, sx, sy = downscale_obs(obs, self.obs_size)
        t = time.perf_counter()
        response, actions = a.predict(instruction, obs)
        if (sx, sy) != (1.0, 1.0) and actions:  # a new list: the agent's history keeps the coordinates of the image it saw
            actions = [rescale_coordinates(c, sx, sy) if isinstance(c, str) else c for c in actions]
        s = a.stats
        return Prediction(response=response or "", actions=actions, cancelled=a.cancelled,
                          seconds=time.perf_counter() - t, t_first_token=s.t_first_token,
                          input_tokens=s.input_tokens, output_tokens=s.output_tokens,
                          tokens_estimated=s.estimated, retries=s.retries, extra={"http": s.http, **({"images_capped": s.images_capped} if s.images_capped else {})})

    def snapshot(self):
        a = self.agent
        return list(a.screenshots), list(a.responses), list(a.actions), a.folded_prefix_k

    def restore(self, snap) -> None:
        a = self.agent
        a.screenshots, a.responses, a.actions = list(snap[0]), list(snap[1]), list(snap[2])
        a.folded_prefix_k = snap[3]
