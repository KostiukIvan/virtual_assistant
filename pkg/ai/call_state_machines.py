# pkg/ai/call_state_machines.py

from transitions import Machine
from transitions.core import MachineError
from collections import defaultdict
from typing import Callable, Dict, List
import threading
import sys


# ---------------------------
# Thread-safe Pub/Sub Bus
# ---------------------------
class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, event_name: str, callback: Callable):
        with self._lock:
            self._subscribers[event_name].append(callback)

    def publish(self, event_name: str, *args, **kwargs):
        # Copy under lock, invoke outside so callbacks can re-enter safely
        with self._lock:
            callbacks = list(self._subscribers[event_name])
        for callback in callbacks:
            callback(*args, **kwargs)


# ---------------------------
# User FSM
# ---------------------------
class UserFSM:
    C_IDLE = "C_IDLE"
    C_SPEAKING = "C_SPEAKING"
    C_SHORT_PAUSE = "C_SHORT_PAUSE"
    C_LONG_PAUSE = "C_LONG_PAUSE"
    C_SILENCE = "C_SILENCE"
    C_LISTENING = "C_LISTENING"
    C_INTERRUPT = "C_INTERRUPT"

    states = [C_IDLE, C_SPEAKING, C_SHORT_PAUSE, C_LONG_PAUSE, C_SILENCE, C_INTERRUPT]

    def __init__(self, bus: EventBus, on_short_pause=None, on_long_pause=None):
        self.bus = bus
        self.on_short_pause = on_short_pause
        self.on_long_pause = on_long_pause
        self.machine = Machine(model=self, states=UserFSM.states, initial=UserFSM.C_IDLE)

       # --- Speaking transitions ---
        self.machine.add_transition("user_starts_speaking", 
            [UserFSM.C_IDLE, UserFSM.C_SHORT_PAUSE, UserFSM.C_LONG_PAUSE, UserFSM.C_SILENCE],
            UserFSM.C_SPEAKING, after=self.notify)
        
        # --- Stop speaking → pause states ---
        self.machine.add_transition("short_pause", UserFSM.C_SPEAKING, UserFSM.C_SHORT_PAUSE, after=["notify", self._fire_short_pause])
        self.machine.add_transition("long_pause", [UserFSM.C_SPEAKING, UserFSM.C_SHORT_PAUSE], UserFSM.C_LONG_PAUSE, after=["notify", self._fire_long_pause])
        self.machine.add_transition("go_silent", "*", UserFSM.C_SILENCE, after=self.notify)

        # --- Listening transitions ---
        self.machine.add_transition("start_listening", "*", UserFSM.C_LISTENING, after=self.notify)
        self.machine.add_transition("stop_listening", UserFSM.C_LISTENING, UserFSM.C_IDLE, after=self.notify)
        
    def is_speaking(self):
        return self.machine.is_state(UserFSM.C_SPEAKING, model=self)

    def notify(self):
        print(f"_{self.state}_", end="")
        sys.stdout.flush()
        self.bus.publish("user_state_changed", state=self.state)


    def _fire_short_pause(self):
        if self.on_short_pause:
            self.on_short_pause()

    def _fire_long_pause(self):
        if self.on_long_pause:
            self.on_long_pause()


# ---------------------------
# Bot FSM (Receiver)
# ---------------------------
class BotFSM:
    R_IDLE = "R_IDLE"
    R_LISTENING = "R_LISTENING"
    R_PROCESSING = "R_PROCESSING"
    R_WAITING_EVENT = "R_WAITING_EVENT"
    R_SPEAKING = "R_SPEAKING"
    R_HOLD = "R_HOLD"
    R_TERMINATE = "R_TERMINATE"

    states = [R_IDLE, R_LISTENING, R_PROCESSING, R_WAITING_EVENT, R_SPEAKING, R_HOLD, R_TERMINATE]

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.machine = Machine(model=self, states=BotFSM.states, initial=BotFSM.R_IDLE)

        # Transitions kept minimal; orchestration will compose them safely.
        self.machine.add_transition("listen", [BotFSM.R_IDLE, BotFSM.R_SPEAKING], BotFSM.R_LISTENING, after=self.notify)
        self.machine.add_transition("process", BotFSM.R_LISTENING, BotFSM.R_PROCESSING, after=self.notify)
        self.machine.add_transition("wait_event", BotFSM.R_PROCESSING, BotFSM.R_WAITING_EVENT, after=self.notify)
        self.machine.add_transition("speak", [BotFSM.R_LISTENING, BotFSM.R_PROCESSING, BotFSM.R_WAITING_EVENT], BotFSM.R_SPEAKING, after=self.notify)
        self.machine.add_transition("hold", BotFSM.R_WAITING_EVENT, BotFSM.R_HOLD, after=self.notify)
        self.machine.add_transition("finish", BotFSM.R_SPEAKING, BotFSM.R_IDLE, after=self.notify)
        self.machine.add_transition("terminate", "*", BotFSM.R_TERMINATE, after=self.notify)
        
    def is_speaking(self):
        return self.machine.is_state(BotFSM.C_SPEAKING, model=self)

    def notify(self):
        print(f"_{self.state}_", end="")
        sys.stdout.flush()
        self.bus.publish("bot_state_changed", state=self.state)


# ---------------------------
# Bot Orchestrator (Safe API)
# ---------------------------
class BotOrchestrator:
    """
    High-level, safe, idempotent commands that move BotFSM using valid sequences.
    This prevents MachineError by routing transitions appropriately.
    """
    def __init__(self, bot: BotFSM):
        self.bot = bot

    def _safe(self, call: Callable):
        try:
            call()
            return True
        except MachineError as e:
            print(f"[BotOrchestrator] Ignored invalid transition from {self.bot.state}: {e}")
            return False

    def ensure_listening(self):
        """
        Move to R_LISTENING from any reasonable state.
        - From R_SPEAKING: listen()
        - From R_IDLE: listen()
        - From R_LISTENING: no-op
        - From R_PROCESSING/R_WAITING_EVENT/R_HOLD: you can decide policy; here we do nothing
          because aborting mid-process may not be desired. Extend as needed.
        """
        if self.bot.state == BotFSM.R_LISTENING:
            return
        if self.bot.state in (BotFSM.R_IDLE, BotFSM.R_SPEAKING):
            self._safe(self.bot.listen)
            
    def set_speaking(self):
        if self.bot.state != BotFSM.R_SPEAKING:
            self._safe(self.bot.speak)
    
    def allowed_to_speak(self):
        return self.bot.state == BotFSM.R_SPEAKING

    def speak_pipeline(self):
        """
        Guarantee a valid path to R_SPEAKING:
        - R_IDLE          → listen → process → speak
        - R_LISTENING     → process → speak
        - R_PROCESSING    → speak
        - R_WAITING_EVENT → speak
        - R_SPEAKING      → no-op
        """
        s = self.bot.state
        if s == BotFSM.R_SPEAKING:
            return
        if s == BotFSM.R_IDLE:
            if not self._safe(self.bot.listen):
                return
            if not self._safe(self.bot.process):
                return
            self._safe(self.bot.speak)
        elif s == BotFSM.R_LISTENING:
            if not self._safe(self.bot.process):
                return
            self._safe(self.bot.speak)
        elif s in (BotFSM.R_PROCESSING, BotFSM.R_WAITING_EVENT):
            self._safe(self.bot.speak)
        else:
            # In HOLD/TERMINATE, do nothing; extend if needed
            pass

    def finish_if_speaking(self):
        if self.bot.state == BotFSM.R_SPEAKING:
            self._safe(self.bot.finish)

    def terminate(self):
        if self.bot.state != BotFSM.R_TERMINATE:
            self._safe(self.bot.terminate)


# ---------------------------
# Wiring & Demo
# ---------------------------
if __name__ == "__main__":
    bus = EventBus()
    user = UserFSM(bus)
    bot = BotFSM(bus)
    botx = BotOrchestrator(bot)

    # Synchronize: user state changes drive bot orchestration
    def on_user_change(state):
        if state == UserFSM.C_SPEAKING:
            botx.ensure_listening()
        elif state == UserFSM.C_LONG_PAUSE:
            # Let the bot take the floor: process → speak (or the right path)
            botx.speak_pipeline()
        elif state == UserFSM.C_SILENCE:
            botx.terminate()
        elif state == UserFSM.C_INTERRUPT:
            # Yield immediately to the user
            botx.ensure_listening()

    bus.subscribe("user_state_changed", on_user_change)

    def on_bot_change(state):
        if state == BotFSM.R_SPEAKING:
            print("Bot is talking, user should wait…")
        elif state == BotFSM.R_IDLE:
            print("Bot finished, user can speak again.")

    bus.subscribe("bot_state_changed", on_bot_change)

    # ---- Simulate conversation (no invalid transitions) ----
    user.user_starts_speaking()  # user starts → bot listens
    user.short_pause()           # user hesitates (no bot change)
    user.long_pause()            # long pause → bot speak pipeline (listen/process/speak as needed)
    botx.finish_if_speaking()    # bot done → R_IDLE
    user.user_starts_speaking()  # user again → bot listens
    user.go_silent()             # silence → terminate