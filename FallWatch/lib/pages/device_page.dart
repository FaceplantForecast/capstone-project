import 'package:flutter/material.dart';
import '../services/device_connection.dart';

class DevicePage extends StatelessWidget {
  DevicePage({super.key});

  final DeviceMonitorService service = DeviceMonitorService();
  
  // form of hh:mm:ss
  String formatDuration(Duration d) {
    final h = d.inHours.toString().padLeft(2, '0');
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$h:$m:$s';
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder(
      stream: service.stateStream,
      builder: (context, _) {
        final deviceList = service.devices.values.toList();
        final connected = service.connected;

        return Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Color(0xFFE8F1FA),
                  Color.fromARGB(255, 187, 179, 179),
                ],
              ),
            ),
            child: SafeArea(
              child: Column(
                children: [
                  // Connection Status
                  AnimatedContainer(
                    duration: const Duration(milliseconds: 300),
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(vertical: 18),
                    color: connected
                      ? const Color.fromARGB(255, 46, 151, 53)
                      : Colors.red.shade600,
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          connected
                            ? Icons.check_circle
                            : Icons.error,
                          color: Colors.white,
                        ),
                        const SizedBox(width: 10),
                        Text(
                          connected
                            ? "Server Connected"
                            : "Server Disconnected",
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 16),

                  // fall status
                  const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 24),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: Text(
                        "Connected Devices",
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.black87,
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 8),

                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      child: deviceList.isEmpty
                        ? const Center(child: Text("No devices connected"))
                        : ListView.builder(
                          itemCount: deviceList.length,
                          itemBuilder: (context, index) {
                            final device = deviceList[index];
                            return _buildDeviceTile(
                              context,
                              device,
                              formatDuration,
                              service,
                            );
                          },
                        ),
                    ),
                  ),
                ],
              ),
            ),
          );
      },
    );
  }
}

Widget _buildDeviceTile(
  BuildContext context, 
  DeviceState device,
  String Function(Duration) formatDuration,
  DeviceMonitorService service,
) {
  String powerLabel;
  if (device.isWallPower) {
    powerLabel = 'Wall Power';
  } else if (device.batteryStartTime != null) {
    final elapsed = DateTime.now().difference(device.batteryStartTime!);
    powerLabel = 'Battery (${formatDuration(elapsed)})';
  } else {
    powerLabel = 'Battery Backup';
  }

  return Padding(
    padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 8),
    child: ExpansionTile(
      tilePadding: const EdgeInsets.symmetric(horizontal: 16),
      collapsedShape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
      ),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
      ),
      backgroundColor: Colors.white,
      collapsedBackgroundColor: Colors.white,
      leading: Icon(
        device.connected ? Icons.check_circle : Icons.error,
        color: device.connected ? Colors.green : Colors.red,
      ),
      title: Text(
        device.label, // uses Display Name, else deviceId
        style: const TextStyle(fontWeight: FontWeight.w600),
      ),
      subtitle: Text(
        device.deviceId,
        style: const TextStyle(fontSize: 12, color: Colors.grey),
      ),
      children: [
        _infoRow("Connection", device.connected ? "Connected" : "Disconnected"),
        _infoRow("Device Status", device.status.label),
        _infoRow("Power", powerLabel),
        _infoRow("Account", device.accountId ?? "Unknown"),
        _infoRow("Last Seen", device.lastSeen?.toString() ?? "N/A",),

      // Renaming
      Padding(
        padding: 
          const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child:
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
          TextButton.icon(
            icon: const Icon(Icons.edit, size: 16),
            label: const Text("Rename"),
            onPressed: () =>
              _showRenameDialog(context, device, service),
          ),
          SizedBox(width: 12),
          TextButton.icon(
            icon: const Icon(Icons.tune, size: 16),
            label: const Text("Recalibrate"),
            onPressed: () {
              service.sendRecalibrationCommand(device.deviceId);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text("Recalibration command sent for ${device.label}"),
                duration: const Duration(seconds: 2),
                )
              );
            },
          ),
          ]
        ),
      ),
    ],
  ),
);
}



void _showRenameDialog(
  BuildContext context,
  DeviceState device,
  DeviceMonitorService service,
) {
  final controller = TextEditingController(text: device.displayName ?? device.deviceId);

  showDialog<void>(
    context: context,
    builder: (ctx) => AlertDialog(
      title: const Text("Rename Device"),
      content: TextField(
        controller: controller,
        decoration: const InputDecoration(
          labelText: "Device Name",
          border: OutlineInputBorder(),
        ),
        autofocus: true,
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text("Cancel"),
        ),
        ElevatedButton(
          onPressed: () {
            service.renameDevice(device.deviceId, controller.text.trim());
            Navigator.pop(context);
          },
          child: const Text("Save"),
        ),
      ],
    ),
  );
}

Widget _infoRow(String label, String value) {
  return ListTile(
    title: Text(label),
    trailing: Text(
      value,
      style: const TextStyle(fontWeight: FontWeight.w500),
    ),
  );
}