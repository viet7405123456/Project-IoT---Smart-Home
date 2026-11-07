"""Sends commands to the relay controller: real ESP32 over HTTP, or a mock
controller for testing before the control board is wired up."""

import requests

COMMAND_TO_REQUEST = {
    "light_on": ("/light", {"state": "on"}),
    "light_off": ("/light", {"state": "off"}),
    "fan_on": ("/fan", {"state": "on"}),
    "fan_off": ("/fan", {"state": "off"}),
    "door_unlock": ("/door", {"state": "unlock"}),
    "door_lock": ("/door", {"state": "lock"}),
}

# command name -> (device key, new value), used to update mock/local state.
COMMAND_TO_STATE = {
    "light_on": ("light", "on"),
    "light_off": ("light", "off"),
    "fan_on": ("fan", "on"),
    "fan_off": ("fan", "off"),
    "door_unlock": ("door", "unlocked"),
    "door_lock": ("door", "locked"),
}


class DeviceController:
    """Sends real HTTP requests to the ESP32 relay controller."""

    def __init__(self, base_url, timeout_seconds=2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.state = {"light": "off", "fan": "off", "door": "locked"}

    def send_command(self, command_name):
        if command_name not in COMMAND_TO_REQUEST:
            return False
        path, params = COMMAND_TO_REQUEST[command_name]
        try:
            response = requests.get(
                f"{self.base_url}{path}", params=params, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            device_key, value = COMMAND_TO_STATE[command_name]
            self.state[device_key] = value
            return True
        except requests.RequestException as exc:
            print(f"[device_controller] Failed to send '{command_name}': {exc}")
            return False

    def get_state(self):
        return self.state


class MockDeviceController:
    """Simulates the relay controller: no network calls, just local state."""

    def __init__(self):
        self.state = {"light": "off", "fan": "off", "door": "locked"}

    def send_command(self, command_name):
        if command_name not in COMMAND_TO_STATE:
            return False
        device_key, value = COMMAND_TO_STATE[command_name]
        self.state[device_key] = value
        print(f"[MOCK] {device_key} -> {value}")
        return True

    def get_state(self):
        return self.state
