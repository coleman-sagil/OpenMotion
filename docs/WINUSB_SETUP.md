# WinUSB Setup for the Leap Motion Controller

The OpenMotion driver talks to the Leap Motion Controller via **libusb**
(through `rusb`). On Windows, libusb needs the **WinUSB** generic driver
installed for the device. The default Windows driver (or the Ultraleap
driver) doesn't expose the low-level USB control/bulk interface that
OpenMotion needs.

## Prerequisites

- **Zadig** — free USB driver installer: https://zadig.akeo.ie/
- The Leap Motion Controller plugged in via USB

## Installation

1. **Stop the Ultraleap tracking service** if installed:
   ```powershell
   Stop-Service UltraleapTracking -Force
   ```
   The service holds the device and must release it before Zadig can see it.

2. **Open Zadig** (run as Administrator)

3. **Select the Leap Motion Controller**:
   - Menu → Options → **List All Devices** (check this)
   - In the dropdown, find the device with:
     - **USB ID**: `F182` : `0003`
     - Name may show as "Leap Motion Controller" or a generic UVC name
   - If multiple entries appear for the same VID/PID, pick **Interface 0**
     (VideoControl — the one that owns the extension units)

4. **Replace the driver**:
   - Set the target driver to **WinUSB** (not libusb-win32, not libusbK)
   - Click **Replace Driver** (or **Install Driver** if no driver is bound)
   - Wait for it to complete (may take 30s–1min)

5. **Repeat for Interface 1** (VideoStreaming):
   - Select the second entry for the same device (Interface 1)
   - Replace with **WinUSB** again

6. **Verify**: run `openleap info` — it should find and claim the device
   without errors.

## What this changes

| Before | After |
|---|---|
| Windows assigns its built-in UVC driver or the Ultraleap driver | WinUSB generic driver bound |
| Ultraleap Gemini tracking service can use the device | Ultraleap service **cannot** use the device |
| OpenMotion driver **cannot** claim the device | OpenMotion driver claims and streams normally |

**You can switch back** at any time by opening Zadig again and replacing
WinUSB with the original driver (shown in the "Current Driver" field), or
by uninstalling the device in Device Manager and letting Windows re-detect
it.

## Caveats

- **Ultraleap's software won't work** while WinUSB is the driver. The two
  stacks (OpenMotion and Ultraleap) are mutually exclusive at the driver
  level. To switch back, use Zadig to restore the original driver.

- **Windows Update may re-bind** the original driver after a system update.
  If `openleap info` starts failing after an update, re-run Zadig.

- **USB 2.0 vs 3.0**: The original LMC is a USB 2.0 device. It works in
  both USB 2.0 and 3.0 ports, but USB 2.0 ports give the full ~30 MB/s
  bandwidth the device expects. Some USB 3.0 hubs introduce latency; if
  you see excessive timeouts, try a direct USB 2.0 port.

- **No elevation needed for streaming** — unlike Linux (which needs root/pkexec
  to detach uvcvideo), WinUSB lets a normal user process claim the device
  once the driver is installed. Zadig itself needs Administrator to install
  the driver.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Device not listed in Zadig | Enable Options → List All Devices; ensure the device is plugged in and the Ultraleap service is stopped |
| `openleap info` says "NoDevice" | Wrong driver or device unplugged. Check Device Manager → Universal Serial Bus devices for the Leap |
| `openleap info` says "Access" or "Busy" | Another process holds the device. Stop the Ultraleap service and any other Leap software |
| Frames are dark / all-black | Device needs IR LEDs enabled. The bring-up sequence handles this; if streaming with `openleap stream`, check the `leds` mask (default `0b101` = left+right) |
